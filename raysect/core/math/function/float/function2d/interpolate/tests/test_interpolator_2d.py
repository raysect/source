
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
Unit tests for array interpolation (not mesh) from within Interpolate2DArray,
including interaction with internal extrapolators.
"""
import unittest
import numpy as np
from raysect.core.math.function.float.function2d.interpolate.interpolator2darray import Interpolator2DArray, \
    id_to_extrapolator, id_to_interpolator
from raysect.core.math.function.float.function2d.interpolate.tests.scripts.generate_2d_splines import X_LOWER, X_UPPER, \
    NB_XSAMPLES, NB_X, X_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN, PRECISION, Y_LOWER, Y_UPPER, NB_YSAMPLES, NB_Y, \
    Y_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MIN, EXTRAPOLATION_RANGE, get_extrapolation_input_values
from raysect.core.math.function.float.function2d.interpolate.tests.data_store.interpololator2d_test_data import \
    TestInterpolatorLoadBigValues, TestInterpolatorLoadNormalValues, TestInterpolatorLoadSmallValues


class TestInterpolators2D(unittest.TestCase):
    def setUp(self) -> None:

        # self.data is a precalculated input values for testing. It's the result of applying function f on self.x
        # as in self.data = f(self.x), where self.x is linearly spaced between X_LOWER and X_UPPER

        #: x and y values used to obtain self.data
        x_in = np.linspace(X_LOWER, X_UPPER, NB_X)
        y_in = np.linspace(Y_LOWER, Y_UPPER, NB_Y)
        self.x = x_in
        self.y = y_in

        self.test_loaded_values = TestInterpolatorLoadNormalValues()
        self.test_loaded_big_values = TestInterpolatorLoadBigValues()
        self.test_loaded_small_values = TestInterpolatorLoadSmallValues()

        #: precalculated result of sampling self.data on self.xsamples
        #   should be set in interpolator specific setup function.
        self.precalc_interpolation = None

        #: x values on which self.precalc_interpolation was samples on
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
        self.ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)

        #: x values on which self.precalc_extrapolation_ arrays were sampled on
        # Extrapolation x and y values
        self.xsamples_out_of_bounds, self.ysamples_out_of_bounds, self.xsamples_in_bounds, self.ysamples_in_bounds = \
            get_extrapolation_input_values(
                X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, X_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN,
                Y_EXTRAP_DELTA_MIN
            )

        #: set precalculated expected extrapolation results  Set in setup_ method
        self.precalc_extrapolation = None

        #: the interpolator object that is being tested. Set in setup_ method
        self.interpolator: Interpolator2DArray = None

    def setup_linear(
            self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool) -> None:
        """
        Sets precalculated values for linear interpolator.
        Called in every test method that addresses linear interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate2DArray object that is being tested. Precalculated interpolation using
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
        self.interpolator = Interpolator2DArray(
            self.x, self.y, self.data, 'linear', extrapolator_type, extrapolation_range, extrapolation_range
        )

    def setup_cubic(self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool):
        """
        Sets precalculated values for cubic interpolator.
        Called in every test method that addresses cubic interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate2DArray object that is being tested.

        WARNING: Generated using a working version to check for differences between versions. Not a mathematical test

        :param extrapolator_type: type of extrapolator 'none', 'linear' or 'cubic'.
        :param extrapolation_range: padding around interpolation range where extrapolation is possible.
        :param big_values: For loading and testing big value saved data.
        :param small_values: For loading and testing small value saved data.
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
        self.interpolator = Interpolator2DArray(
            self.x, self.y, self.data, 'cubic', extrapolator_type, extrapolation_range, extrapolation_range
        )

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
        for i in range(len(self.xsamples_in_bounds)):
            self.assertRaises(
                ValueError, self.interpolator, **{'x': self.xsamples_in_bounds[i], 'y': self.ysamples_in_bounds[i]}
            )

    def test_linear_interpolation_extrapolators(self):
        no_test_for_extrapolator = ['linear']
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_linear(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False)
            if extrapolator_type != 'none':
                if extrapolator_type not in no_test_for_extrapolator:
                    if extrapolator_type == 'nearest':
                        gradient_continuity = False
                    else:
                        gradient_continuity = True
                    self.run_general_extrapolation_tests(
                        gradient_continuity=gradient_continuity, extrapolator_type=extrapolator_type
                    )
            self.run_general_interpolation_tests()

        # Tests for big values
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_linear(extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False)
            if extrapolator_type != 'none':
                if extrapolator_type not in no_test_for_extrapolator:
                    if extrapolator_type == 'nearest':
                        gradient_continuity = False
                    else:
                        gradient_continuity = True
                    self.run_general_extrapolation_tests(
                        gradient_continuity=gradient_continuity, extrapolator_type=extrapolator_type
                    )
            self.run_general_interpolation_tests()

        # Tests for small values
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_linear(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True)
            if extrapolator_type != 'none':
                if extrapolator_type not in no_test_for_extrapolator:
                    if extrapolator_type == 'nearest':
                        gradient_continuity = False
                    else:
                        gradient_continuity = True
                    self.run_general_extrapolation_tests(
                        gradient_continuity=gradient_continuity, extrapolator_type=extrapolator_type
                    )

            self.run_general_interpolation_tests()

    def test_cubic_interpolation_extrapolators(self):
        """
        Testing against a previous version of cubic interpolators to highlight changes.
        """
        test_on = True
        # Temporarily turn off cubic 2D tests
        if test_on:
            for extrapolator_type in id_to_extrapolator.keys():
                self.setup_cubic(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False)
                if extrapolator_type != 'none':
                    if extrapolator_type == 'nearest':
                        gradient_continuity = False
                    else:
                        gradient_continuity = True
                    self.run_general_extrapolation_tests(
                        gradient_continuity=gradient_continuity, extrapolator_type=extrapolator_type
                    )
                self.run_general_interpolation_tests()

            # Tests for big values
            for extrapolator_type in id_to_extrapolator.keys():
                self.setup_cubic(extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False)
                if extrapolator_type != 'none':
                    if extrapolator_type == 'nearest':
                        gradient_continuity = False
                    else:
                        gradient_continuity = True
                    self.run_general_extrapolation_tests(
                        gradient_continuity=gradient_continuity, extrapolator_type=extrapolator_type
                    )
                self.run_general_interpolation_tests()

            # Tests for small values
            for extrapolator_type in id_to_extrapolator.keys():
                self.setup_cubic(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True)
                if extrapolator_type != 'none':
                    if extrapolator_type == 'nearest':
                        gradient_continuity = False
                    else:
                        gradient_continuity = True
                    self.run_general_extrapolation_tests(
                        gradient_continuity=gradient_continuity, extrapolator_type=extrapolator_type
                    )

                self.run_general_interpolation_tests()

    def run_general_extrapolation_tests(self, gradient_continuity=True, extrapolator_type='', significant_tolerance=None):
        # Test extrapolator out of range, there should be an error raised
        for i in range(len(self.xsamples_out_of_bounds)):
            dict_kwargs_extrapolator_call = {
                'x': self.xsamples_out_of_bounds[i], 'y': self.ysamples_out_of_bounds[i]
            }
            self.assertRaises(
                ValueError, self.interpolator, **dict_kwargs_extrapolator_call
            )

        # Test extrapolation inside extrapolation range matches the predefined values
        for i in range(len(self.xsamples_in_bounds)):
            delta_max = np.abs(self.precalc_extrapolation[i]/np.power(10., PRECISION - 1))
            self.assertAlmostEqual(
                self.interpolator(
                    self.xsamples_in_bounds[i], self.ysamples_in_bounds[i]), self.precalc_extrapolation[i],
                delta=delta_max, msg='Failed for ' + extrapolator_type + f'{self.xsamples_in_bounds[i]} and '
                                                                         f'{self.ysamples_in_bounds[i]}'
            )

        # Turned off gradient testing for now
        test_on = False
        if test_on:
            # Test gradient continuity between interpolation and extrapolation
            delta_max_lower = np.abs(self.precalc_interpolation[0] / np.power(10., PRECISION - 1))
            delta_max_upper = np.abs(self.precalc_interpolation[-1] / np.power(10., PRECISION - 1))
            if gradient_continuity:
                gradients_lower, gradients_upper = self.interpolator.test_edge_gradients()
                self.assertAlmostEqual(gradients_lower[0], gradients_lower[1], delta=delta_max_lower)
                self.assertAlmostEqual(gradients_upper[0], gradients_upper[1], delta=delta_max_upper)

    def run_general_interpolation_tests(self, significant_tolerance=None):
        # Test interpolation against xsample
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                delta_max = np.abs(self.precalc_interpolation[i, j] / np.power(10., PRECISION - 1))
                self.assertAlmostEqual(
                    self.interpolator(self.xsamples[i], self.ysamples[j]), self.precalc_interpolation[i, j],
                    delta=delta_max
                )

    def initialise_tests_on_interpolators(self, x_values, y_values, f_values):
        # Test for all combinations
        for extrapolator_type in id_to_extrapolator.keys():
            for interpolator_type in id_to_interpolator.keys():
                dict_kwargs_interpolators = {
                    'x': x_values, 'y': y_values, 'f': f_values, 'interpolation_type': interpolator_type,
                    'extrapolation_type': extrapolator_type, 'extrapolation_range_x': 2.0, 'extrapolation_range_y': 2.0
                }
                self.assertRaises(ValueError, Interpolator2DArray, **dict_kwargs_interpolators)

    def test_initialisation_errors(self):
        # monotonicity x
        x_wrong = np.copy(self.x)
        x_wrong[0] = self.x[1]
        x_wrong[1] = self.x[0]
        self.initialise_tests_on_interpolators(x_wrong, self.y, self.test_loaded_values.data)

        # monotonicity y
        y_wrong = np.copy(self.y)
        y_wrong[0] = self.y[1]
        y_wrong[1] = self.y[0]
        self.initialise_tests_on_interpolators(self.x, y_wrong, self.test_loaded_values.data)

        # test repeated coordinate x
        x_wrong = np.copy(self.x)
        x_wrong[0] = x_wrong[1]
        self.initialise_tests_on_interpolators(x_wrong, self.y, self.test_loaded_values.data)

        # test repeated coordinate y
        y_wrong = np.copy(self.y)
        y_wrong[0] = y_wrong[1]
        self.initialise_tests_on_interpolators(self.x, y_wrong, self.test_loaded_values.data)

        # mismatch array size between x and data
        x_wrong = np.copy(self.x)
        x_wrong = x_wrong[:-1]
        self.initialise_tests_on_interpolators(x_wrong, self.y, self.test_loaded_values.data)

        # mismatch array size between y and data
        y_wrong = np.copy(self.y)
        y_wrong = y_wrong[:-1]
        self.initialise_tests_on_interpolators(self.x, y_wrong, self.test_loaded_values.data)

        # Test array length 1, Arrays are too short
        x_wrong = np.copy(self.x)
        y_wrong = np.copy(self.y)
        f_wrong = np.copy(self.test_loaded_values.data)
        x_wrong = x_wrong[0]
        y_wrong = y_wrong[0]
        f_wrong = f_wrong[0, 0]
        self.initialise_tests_on_interpolators(x_wrong, y_wrong, f_wrong)

        # Incorrect dimension (1D data)
        x_wrong = np.array(np.concatenate((np.copy(self.x)[:, np.newaxis], np.copy(self.x)[:, np.newaxis]), axis=1))
        f_wrong = np.array(np.concatenate((
            np.copy(self.test_loaded_values.data)[:, np.newaxis], np.copy(self.test_loaded_values.data)[:, np.newaxis]
        ), axis=1)
        )
        self.initialise_tests_on_interpolators(x_wrong, y_wrong, f_wrong)
