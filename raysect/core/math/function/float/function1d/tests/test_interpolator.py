
# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
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
Unit tests for the Interpolator1DCubic class from within Interpolator1DArray,
including interaction with Extrapolator1DLinear and Extrapolator1DNearest.
"""
import unittest
import numpy as np
from raysect.core.math.function.float.function1d.interpolate import Interpolator1DArray, id_to_extrapolator, \
    id_to_interpolator, permitted_interpolation_combinations
from raysect.core.math.function.float.function1d.tests.data.interpolator1d_test_data import \
    TestInterpolatorLoadBigValuesUneven, TestInterpolatorLoadNormalValuesUneven, TestInterpolatorLoadSmallValuesUneven,\
    TestInterpolatorLoadBigValues, TestInterpolatorLoadNormalValues, TestInterpolatorLoadSmallValues


def large_extrapolation_range(xsamples_in, extrapolation_range, n_extrap):
    x_lower = np.linspace(xsamples_in[0] - extrapolation_range, xsamples_in[0], n_extrap + 1)[:-1]
    x_upper = np.linspace(xsamples_in[-1], xsamples_in[-1] + extrapolation_range, n_extrap + 1)[1:]

    xsamples_in_expanded = np.concatenate((x_lower, xsamples_in, x_upper), axis=0)
    edge_start_x = np.arange(0, n_extrap, 1, dtype=int)
    edge_end_x = np.arange(len(xsamples_in_expanded) - 1, len(xsamples_in_expanded) - 1 - n_extrap, -1, dtype=int)
    edge_indicies_x = np.concatenate((edge_start_x, edge_end_x), axis=0)

    xsamples_extrap_in_bounds = []
    for i_x in range(len(xsamples_in_expanded)):
        if not (i_x not in edge_indicies_x):
            xsamples_extrap_in_bounds.append(xsamples_in_expanded[i_x])

    return np.array(xsamples_extrap_in_bounds)


def extrapolation_out_of_bound_points(x_lower, x_upper, x_extrap_delta_max, extrapolation_range):
    xsamples_extrap_out_of_bounds_options = np.array(
        [x_lower - extrapolation_range - x_extrap_delta_max, (x_lower + x_upper) / 2.,
         x_upper + extrapolation_range + x_extrap_delta_max])

    xsamples_extrap_out_of_bounds = []
    edge_indicies = [0, len(xsamples_extrap_out_of_bounds_options) - 1]
    for i_x in range(len(xsamples_extrap_out_of_bounds_options)):
        if i_x in edge_indicies:
            xsamples_extrap_out_of_bounds.append(xsamples_extrap_out_of_bounds_options[i_x])

    return np.array(xsamples_extrap_out_of_bounds)


def uneven_linspace(x_lower, x_upper, n_2, offset_fraction):
    dx = (x_upper - x_lower)/(n_2 - 1)
    offset_x = offset_fraction * dx
    x1 = np.linspace(x_lower, x_upper, NB_X)
    x2 = np.linspace(x_lower + offset_x, x_upper + offset_x, n_2)[:-1]
    return np.sort(np.concatenate((x1, x2), axis=0))


X_LOWER = 0.0
X_UPPER = 1.0
X_EXTRAP_DELTA_MAX = 0.08
X_EXTRAP_DELTA_MIN = 0.04

NB_X = 10
NB_XSAMPLES = 30

EXTRAPOLATION_RANGE = 2.0
N_EXTRAPOLATION = 3
PRECISION = 12

BIG_VALUE_FACTOR = 20.
SMALL_VALUE_FACTOR = -20.


class TestInterpolators1D(unittest.TestCase):
    """
    Testing class for 1D interpolators and extrapolators.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # data is a precalculated input values for testing. It's the result of applying function f on self.x.
        # as in data = f(self.x), where self.x is linearly spaced between X_LOWER and X_UPPER.

        #: x values used to obtain data.
        self.x = np.linspace(X_LOWER, X_UPPER, NB_X)
        self.x_uneven = uneven_linspace(X_LOWER, X_UPPER, NB_X, offset_fraction=1./3.)
        self.reference_loaded_values = TestInterpolatorLoadNormalValues()
        self.reference_loaded_big_values = TestInterpolatorLoadBigValues()
        self.reference_loaded_small_values = TestInterpolatorLoadSmallValues()

        self.reference_loaded_values_uneven = TestInterpolatorLoadNormalValuesUneven()
        self.reference_loaded_big_values_uneven = TestInterpolatorLoadBigValuesUneven()
        self.reference_loaded_small_values_uneven = TestInterpolatorLoadSmallValuesUneven()

        #: x values on which interpolation_data were sampled on.
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)

        #: x values on which extrapolation_data arrays were sampled on.
        #: x, y values on which extrapolation_data arrays were sampled on.
        # Extrapolation x and y values.
        self.xsamples_out_of_bounds = extrapolation_out_of_bound_points(
                X_LOWER, X_UPPER, X_EXTRAP_DELTA_MAX, EXTRAPOLATION_RANGE
        )
        self.xsamples_in_bounds = large_extrapolation_range(self.xsamples, EXTRAPOLATION_RANGE, N_EXTRAPOLATION)

    def setup_linear(
            self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool,
            uneven_spacing: bool):
        """
        Sets precalculated values for linear interpolator.
        Called in every test method that addresses linear interpolation.

        interpolator will hold Interpolator1DArray object that is being tested. Precalculated interpolation using
        scipy.interpolate.interp1d(kind=linear), generated using scipy version 1.6.3.

        :param extrapolator_type: type of extrapolator 'none' or 'linear'.
        :param extrapolation_range: padding around interpolation range where extrapolation is possible.
        :param big_values: For loading and testing big value saved data.
        :param small_values: For loading and testing small value saved data.
        :param uneven_spacing: For unevenly spaced test data.
        """

        # Set precalculated expected interpolation results  using scipy.interpolate.interp1d(kind=linear).
        # This is the result of sampling data on self.xsamples.

        if uneven_spacing:

            if big_values:
                self.value_storage_obj = self.reference_loaded_big_values_uneven

            elif small_values:
                self.value_storage_obj = self.reference_loaded_small_values_uneven

            else:
                self.value_storage_obj = self.reference_loaded_values_uneven

        else:

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
        if uneven_spacing:
            interpolator = Interpolator1DArray(self.x_uneven, data, 'linear', extrapolator_type, extrapolation_range)

        else:
            interpolator = Interpolator1DArray(self.x, data, 'linear', extrapolator_type, extrapolation_range)

        return interpolator, interpolation_data, extrapolation_data

    def setup_cubic(
            self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool,
            uneven_spacing: bool):
        """
        Sets precalculated values for cubic interpolator.
        Called in every test method that addresses cubic interpolation.

        interpolator will hold Interpolator1DArray object that is being tested. Generated using scipy
        version 1.6.3 scipy.interpolate.CubicHermiteSpline, with input gradients at the spline knots.

        :param extrapolator_type: type of extrapolator 'none' or 'linear'.
        :param extrapolation_range: padding around interpolation range where extrapolation is possible.
        :param big_values: For loading and testing big value saved data.
        :param small_values: For loading and testing small value saved data.
        :param uneven_spacing: For unevenly spaced test data.
        """

        # Set precalculated expected interpolation results.
        # This is the result of sampling data on self.xsamples.
        if uneven_spacing:

            if big_values:
                self.value_storage_obj = self.reference_loaded_big_values_uneven

            elif small_values:
                self.value_storage_obj = self.reference_loaded_small_values_uneven

            else:
                self.value_storage_obj = self.reference_loaded_values_uneven

        else:

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
        if uneven_spacing:
            interpolator = Interpolator1DArray(self.x_uneven, data, 'cubic', extrapolator_type, extrapolation_range)

        else:
            interpolator = Interpolator1DArray(self.x, data, 'cubic', extrapolator_type, extrapolation_range)

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
            raise ValueError(f'Extrapolation type {extrapolator_type} not found or no test. options are {id_to_extrapolator.keys()}')

        return extrapolation_data

    def test_extrapolation_none(self):
        """
        Testing that extrapolator_type 'none' returns a ValueError rather than data when attempting to extrapolate
        outside its extrapolation range.
        """
        interpolator, _, _ = self.setup_linear(
            'none', EXTRAPOLATION_RANGE, big_values=False, small_values=False, uneven_spacing=False
        )
        for i in range(len(self.xsamples_in_bounds)):
            with self.assertRaises(
                    ValueError, msg=f'No ValueError raised when testing extrapolator type none, at point '
                                    f'x ={self.xsamples_in_bounds[i]} that should be '
                                    f'outside of the interpolator range of {self.x[0]}<=x<={self.x[-1]}'):
                interpolator(self.xsamples_in_bounds[i])

    def test_linear_interpolation_extrapolators(self):
        """
        Testing against linear interpolator objects for interpolation and extrapolation agreement.

        Testing for linear and nearest extrapolation are compared with simple external functions in
        raysect.core.math.function.function1d.tests.scripts.generate_1d_splines. interp1d(kind='linear') was used
        for linear interpolation.
        """
        for uneven_spacing in [True, False]:
            if uneven_spacing:
                uneven_spacing_str = 'uneven spacing'

            else:
                uneven_spacing_str = 'even spacing'

            for extrapolator_type in id_to_extrapolator.keys():
                if extrapolator_type in permitted_interpolation_combinations['linear']:
                    interpolator, interpolation_data, extrapolation_data = self.setup_linear(
                        extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False,
                        uneven_spacing=uneven_spacing
                    )

                    if extrapolator_type != 'none':
                        self.run_general_extrapolation_tests(
                            interpolator, extrapolation_data, extrapolator_type=extrapolator_type,
                            interpolator_str='linear values ' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, extrapolator_type=extrapolator_type,
                        interpolator_str='linear values ' + uneven_spacing_str
                    )

                else:
                    with self.assertRaises(ValueError, msg=f'linear interpolation and {extrapolator_type} extrapolation'
                                                           f'are not compatible types, yet no ValueError was raised'
                                                           f'setting of types are found in '
                                                           f'permitted_interpolation_combinations'):
                        self.setup_linear(
                            extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False,
                            uneven_spacing=uneven_spacing
                        )

            # Tests for big values
            for extrapolator_type in id_to_extrapolator.keys():
                if extrapolator_type in permitted_interpolation_combinations['linear']:
                    interpolator, interpolation_data, extrapolation_data = self.setup_linear(
                        extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False,
                        uneven_spacing=uneven_spacing
                    )

                    if extrapolator_type != 'none':
                        self.run_general_extrapolation_tests(
                            interpolator, extrapolation_data, extrapolator_type=extrapolator_type,
                            interpolator_str='linear big values ' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, extrapolator_type=extrapolator_type,
                        interpolator_str='linear big values ' + uneven_spacing_str
                    )

                else:
                    with self.assertRaises(ValueError, msg=f'linear interpolation and {extrapolator_type} extrapolation'
                                                           f'are not compatible types, yet no ValueError was raised'
                                                           f'setting of types are found in '
                                                           f'permitted_interpolation_combinations'):
                        self.setup_linear(
                            extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False,
                            uneven_spacing=uneven_spacing
                        )

            # Tests for small values
            for extrapolator_type in id_to_extrapolator.keys():
                if extrapolator_type in permitted_interpolation_combinations['linear']:
                    interpolator, interpolation_data, extrapolation_data = self.setup_linear(
                        extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True,
                        uneven_spacing=uneven_spacing
                    )

                    if extrapolator_type != 'none':
                        self.run_general_extrapolation_tests(
                            interpolator, extrapolation_data, extrapolator_type=extrapolator_type,
                            interpolator_str='linear small values ' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, extrapolator_type=extrapolator_type,
                        interpolator_str='linear small values ' + uneven_spacing_str
                    )

                else:
                    with self.assertRaises(ValueError, msg=f'linear interpolation and {extrapolator_type} extrapolation'
                                                           f'are not compatible types, yet no ValueError was raised'
                                                           f'setting of types are found in '
                                                           f'permitted_interpolation_combinations'):
                        self.setup_linear(
                            extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True,
                            uneven_spacing=uneven_spacing
                        )

    def test_cubic_interpolation_extrapolators(self):
        """
        Testing against scipy.interpolate.CubicHermiteSpline with the same gradient calculations. For uneven spacing,
        the function is tested against itself and is not a mathematical test.
        """
        for uneven_spacing in [True, False]:
            if uneven_spacing:
                uneven_spacing_str = 'uneven spacing'

            else:
                uneven_spacing_str = 'even spacing'

            for extrapolator_type in id_to_extrapolator.keys():
                if extrapolator_type in permitted_interpolation_combinations['cubic']:
                    interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                        extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False,
                        uneven_spacing=uneven_spacing
                    )

                    if extrapolator_type != 'none':
                        self.run_general_extrapolation_tests(
                            interpolator, extrapolation_data, extrapolator_type=extrapolator_type,
                            interpolator_str='cubic values ' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, extrapolator_type=extrapolator_type,
                        interpolator_str='cubic values ' + uneven_spacing_str
                    )

                else:
                    with self.assertRaises(ValueError, msg=f'cubic interpolation and {extrapolator_type} extrapolation'
                                                           f'are not compatible types, yet no ValueError was raised'
                                                           f'setting of types are found in '
                                                           f'permitted_interpolation_combinations'):
                        self.setup_cubic(
                            extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False,
                            uneven_spacing=uneven_spacing_str
                        )

            # Tests for big values
            for extrapolator_type in id_to_extrapolator.keys():
                if extrapolator_type in permitted_interpolation_combinations['cubic']:
                    interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                        extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False,
                        uneven_spacing=uneven_spacing_str
                    )

                    if extrapolator_type != 'none':
                        self.run_general_extrapolation_tests(
                            interpolator, extrapolation_data, extrapolator_type=extrapolator_type,
                            interpolator_str='cubic big values ' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, extrapolator_type=extrapolator_type,
                        interpolator_str='cubic big values ' + uneven_spacing_str
                    )

                else:
                    with self.assertRaises(ValueError, msg=f'cubic interpolation and {extrapolator_type} extrapolation'
                                                           f'are not compatible types, yet no ValueError was raised'
                                                           f'setting of types are found in '
                                                           f'permitted_interpolation_combinations'):
                        self.setup_cubic(
                            extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False,
                            uneven_spacing=uneven_spacing_str
                        )

            # Tests for small values
            for extrapolator_type in id_to_extrapolator.keys():
                if extrapolator_type in permitted_interpolation_combinations['cubic']:
                    interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                        extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True,
                        uneven_spacing=uneven_spacing_str
                    )

                    if extrapolator_type != 'none':
                        self.run_general_extrapolation_tests(
                            interpolator, extrapolation_data, extrapolator_type=extrapolator_type,
                            interpolator_str='cubic small values ' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, extrapolator_type=extrapolator_type,
                        interpolator_str='cubic small values ' + uneven_spacing_str
                    )

                else:
                    with self.assertRaises(ValueError, msg=f'cubic interpolation and {extrapolator_type} extrapolation'
                                                           f'are not compatible types, yet no ValueError was raised'
                                                           f'setting of types are found in '
                                                           f'permitted_interpolation_combinations'):
                        self.setup_cubic(
                            extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True,
                            uneven_spacing=uneven_spacing_str
                        )

    def run_general_extrapolation_tests(self, interpolator, extrapolation_data, extrapolator_type='',
                                        interpolator_str=''):
        # Test extrapolator out of range, there should be an error raised.
        for i in range(len(self.xsamples_out_of_bounds)):
            with self.assertRaises(
                    ValueError, msg=f'No ValueError raised when testing interpolator type {interpolator_str} '
                                    f'extrapolator type {extrapolator_type}, at point x ='
                                    f'{self.xsamples_out_of_bounds[i]} that should be outside of the interpolator range'
                                    f' of {self.x[0]}<=x<={self.x[-1]} and also outside of the extrapolation range '
                                    f'{EXTRAPOLATION_RANGE} from these edges.'):
                interpolator(self.xsamples_out_of_bounds[i])

        # Test extrapolation inside extrapolation range matches the predefined values.
        for i in range(len(self.xsamples_in_bounds)):
            delta_max = np.abs(extrapolation_data[i]/np.power(10., PRECISION - 1))
            self.assertAlmostEqual(
                interpolator(self.xsamples_in_bounds[i]), extrapolation_data[i],
                delta=delta_max, msg=f'Failed for interpolator {interpolator_str} with extrapolator {extrapolator_type}'
                                     f', attempting to extrapolate at point x ={self.xsamples_in_bounds[i]} that '
                                     f'should be outside of the interpolator range of {self.x[0]}<=x<={self.x[-1]} and '
                                     f'inside the extrapolation range {EXTRAPOLATION_RANGE} from these edges.'
            )

    def run_general_interpolation_tests(self, interpolator, interpolation_data, extrapolator_type='',
                                        interpolator_str=''):
        # Test interpolation against xsample
        for i in range(len(self.xsamples)):
            delta_max = np.abs(interpolation_data[i] / np.power(10., PRECISION - 1))
            self.assertAlmostEqual(
                interpolator(self.xsamples[i]), interpolation_data[i], delta=delta_max,
                msg=f'Failed for interpolator {interpolator_str} with extrapolator {extrapolator_type}, attempting to '
                    f'interpolate at point x ={self.xsamples[i]} that should be inside of the interpolator range of '
                    f'{self.x[0]}<=x<={self.x[-1]}.'
            )

    def initialise_tests_on_interpolators(self, x_values, f_values, problem_str=''):
        # Test for all combinations
        for extrapolator_type in id_to_extrapolator.keys():
            for interpolator_type in id_to_interpolator.keys():
                with self.assertRaises(
                        ValueError, msg=f'No ValueError raised when testing interpolator type {interpolator_type} '
                                        f'extrapolator type {extrapolator_type}, trying to initialise a test with '
                                        f'incorrect {problem_str}.'):
                    Interpolator1DArray(
                        x=x_values, f=f_values, interpolation_type=interpolator_type,
                        extrapolation_type=extrapolator_type, extrapolation_range=2.0
                    )

    def test_incorrect_spline_knots(self):
        """
        Test for bad data being supplied to the interpolators, x inputs must be increasing.

        Test x monotonically increases, test if an x spline knots have repeated values.

        """
        # monotonicity
        x_wrong = np.copy(self.x)
        x_wrong[0] = self.x[1]
        x_wrong[1] = self.x[0]
        self.initialise_tests_on_interpolators(
            x_wrong, self.reference_loaded_values.data,
            problem_str='monotonicity with the first and second x spline knot the wrong way around'
        )

        # test repeated coordinate
        x_wrong = np.copy(self.x)
        x_wrong[0] = x_wrong[1]
        self.initialise_tests_on_interpolators(
            x_wrong, self.reference_loaded_values.data,
            problem_str='the first spline knot is a repeat of the second x spline knot'
        )

        # mismatch array size between x and data
        x_wrong = np.copy(self.x)
        x_wrong = x_wrong[:-1]
        self.initialise_tests_on_interpolators(
            x_wrong, self.reference_loaded_values.data, problem_str='the last x spline knot has been removed'
        )

    def test_incorrect_array_length(self):
        """
        Make array inputs have length 1 (too short) in 1 or more dimensions. Then check for a ValueError.
        """
        # Test array length 1
        # Arrays are too short
        x_wrong = np.copy(self.x)
        f_wrong = np.copy(self.reference_loaded_values.data)
        x_wrong = np.array([x_wrong[0]])
        f_wrong = np.array([f_wrong[0]])

        self.initialise_tests_on_interpolators(
            x_wrong, f_wrong, problem_str='there is only 1 (x, f) spline knot, which is not enough knots'
        )

    def test_incorrect_array_dimension_combination(self):
        """
        Make array inputs have higher dimensions. Then check for a ValueError.
        """
        # Incorrect dimension (2D data)
        x_wrong = np.array(np.concatenate((np.copy(self.x)[:, np.newaxis], np.copy(self.x)[:, np.newaxis]), axis=1))
        f_wrong = np.array(np.concatenate((
            np.copy(self.reference_loaded_values.data)[:, np.newaxis],
            np.copy(self.reference_loaded_values.data)[:, np.newaxis]
        ), axis=1)
        )
        self.initialise_tests_on_interpolators(
            x_wrong, f_wrong, problem_str='accidentally supplying 2D data'
        )
