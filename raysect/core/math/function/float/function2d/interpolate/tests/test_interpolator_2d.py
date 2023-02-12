
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
Unit tests for array interpolation (not mesh) from within Interpolate2DArray,
including interaction with internal extrapolators.
"""
import unittest
import numpy as np
from raysect.core.math.function.float.function2d.interpolate.interpolator2darray import Interpolator2DArray, \
    id_to_extrapolator, id_to_interpolator, permitted_interpolation_combinations
from raysect.core.math.function.float.function2d.interpolate.tests.data.interpolator2d_test_data import \
    TestInterpolatorLoadBigValues, TestInterpolatorLoadNormalValues, TestInterpolatorLoadSmallValues, \
    TestInterpolatorLoadBigValuesUneven, TestInterpolatorLoadNormalValuesUneven, TestInterpolatorLoadSmallValuesUneven


X_LOWER = -1.0
X_UPPER = 1.0
Y_LOWER = -1.0
Y_UPPER = 1.0
Y_EXTRAP_DELTA_MAX = 0.08
X_EXTRAP_DELTA_MAX = 0.08

NB_X = 10
NB_Y = 10
NB_XSAMPLES = 13
NB_YSAMPLES = 13

N_EXTRAPOLATION = 3

EXTRAPOLATION_RANGE = 2.0

PRECISION = 12


def extrapolation_out_of_bound_points(x_lower, x_upper, y_lower, y_upper, x_extrap_delta_max, y_extrap_delta_max,
                                      extrapolation_range):
    xsamples_extrap_out_of_bounds_options = np.array(
        [x_lower - extrapolation_range - x_extrap_delta_max, (x_lower + x_upper) / 2.,
         x_upper + extrapolation_range + x_extrap_delta_max])

    ysamples_extrap_out_of_bounds_options = np.array(
        [y_lower - extrapolation_range - y_extrap_delta_max, (y_lower + y_upper) / 2.,
         y_upper + extrapolation_range + y_extrap_delta_max])

    xsamples_extrap_out_of_bounds = []
    ysamples_extrap_out_of_bounds = []
    edge_indicies_x = [0, len(xsamples_extrap_out_of_bounds_options) - 1]
    edge_indicies_y = [0, len(ysamples_extrap_out_of_bounds_options) - 1]
    for i_x in range(len(xsamples_extrap_out_of_bounds_options)):
        for j_y in range(len(ysamples_extrap_out_of_bounds_options)):
            if not (i_x not in edge_indicies_x and j_y not in edge_indicies_y):
                xsamples_extrap_out_of_bounds.append(xsamples_extrap_out_of_bounds_options[i_x])
                ysamples_extrap_out_of_bounds.append(ysamples_extrap_out_of_bounds_options[j_y])
    return np.array(xsamples_extrap_out_of_bounds), np.array(ysamples_extrap_out_of_bounds)


def uneven_linspace(x_lower, x_upper, n_2, offset_fraction):
    dx = (x_upper - x_lower)/(n_2 - 1)
    offset_x = offset_fraction * dx
    x1 = np.linspace(x_lower, x_upper, NB_X)
    x2 = np.linspace(x_lower + offset_x, x_upper + offset_x, n_2)[:-1]
    return np.sort(np.concatenate((x1, x2), axis=0))


def large_extrapolation_range(xsamples_in, ysamples_in, extrapolation_range, n_extrap):
    x_lower = np.linspace(xsamples_in[0] - extrapolation_range, xsamples_in[0], n_extrap + 1)[:-1]
    x_upper = np.linspace(xsamples_in[-1], xsamples_in[-1] + extrapolation_range, n_extrap + 1)[1:]
    y_lower = np.linspace(ysamples_in[0] - extrapolation_range, ysamples_in[0], n_extrap + 1)[:-1]
    y_upper = np.linspace(ysamples_in[-1], ysamples_in[-1] + extrapolation_range, n_extrap + 1)[1:]

    xsamples_in_expanded = np.concatenate((x_lower, xsamples_in, x_upper), axis=0)
    ysamples_in_expanded = np.concatenate((y_lower, ysamples_in, y_upper), axis=0)
    edge_start_x = np.arange(0, n_extrap, 1, dtype=int)
    edge_end_x = np.arange(len(xsamples_in_expanded) - 1, len(xsamples_in_expanded) - 1 - n_extrap, -1, dtype=int)
    edge_start_y = np.arange(0, n_extrap, 1, dtype=int)
    edge_end_y = np.arange(len(ysamples_in_expanded) - 1, len(ysamples_in_expanded) - 1 - n_extrap, -1, dtype=int)
    edge_indicies_x = np.concatenate((edge_start_x, edge_end_x), axis=0)
    edge_indicies_y = np.concatenate((edge_start_y, edge_end_y), axis=0)

    xsamples_extrap_in_bounds = []
    ysamples_extrap_in_bounds = []
    for i_x in range(len(xsamples_in_expanded)):
        for j_y in range(len(ysamples_in_expanded)):
            if not (i_x not in edge_indicies_x and j_y not in edge_indicies_y):
                xsamples_extrap_in_bounds.append(xsamples_in_expanded[i_x])
                ysamples_extrap_in_bounds.append(ysamples_in_expanded[j_y])
    return np.array(xsamples_extrap_in_bounds), np.array(ysamples_extrap_in_bounds)


class TestInterpolators2D(unittest.TestCase):
    """
    Testing class for 2D interpolators and extrapolators.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # data is a precalculated input values for testing. It's the result of applying function f on self.x
        # as in data = f(self.x, self.y), where self.x, self.y are linearly spaced between X_LOWER and X_UPPER ...

        #: x and y values used to obtain data
        x_in = np.linspace(X_LOWER, X_UPPER, NB_X)
        y_in = np.linspace(Y_LOWER, Y_UPPER, NB_Y)
        self.x = x_in
        self.y = y_in
        self.x_uneven = uneven_linspace(X_LOWER, X_UPPER, NB_X, offset_fraction=1. / 3.)
        self.y_uneven = uneven_linspace(Y_LOWER, Y_UPPER, NB_Y, offset_fraction=1. / 3.)

        self.reference_loaded_values = TestInterpolatorLoadNormalValues()
        self.reference_loaded_big_values = TestInterpolatorLoadBigValues()
        self.reference_loaded_small_values = TestInterpolatorLoadSmallValues()

        self.reference_loaded_values_uneven = TestInterpolatorLoadNormalValuesUneven()
        self.reference_loaded_big_values_uneven = TestInterpolatorLoadBigValuesUneven()
        self.reference_loaded_small_values_uneven = TestInterpolatorLoadSmallValuesUneven()

        #: x values on which interpolation_data was samples on.
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
        self.ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)

        #: x, y values on which extrapolation_data arrays were sampled on.
        # Extrapolation x and y values.
        self.xsamples_out_of_bounds, self.ysamples_out_of_bounds = extrapolation_out_of_bound_points(
            X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, X_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MAX, EXTRAPOLATION_RANGE
        )
        self.xsamples_in_bounds, self.ysamples_in_bounds = large_extrapolation_range(
            self.xsamples, self.ysamples, EXTRAPOLATION_RANGE, N_EXTRAPOLATION
        )

    def setup_linear(self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool,
                     uneven_spacing: bool):
        """
        Sets precalculated values for linear interpolator.
        Called in every test method that addresses linear interpolation.

        interpolator will hold Interpolate2DArray object that is being tested.

        :param extrapolator_type: type of extrapolator 'none' or 'linear'.
        :param extrapolation_range: padding around interpolation range where extrapolation is possible.
        :param big_values: For loading and testing big value saved data.
        :param small_values: For loading and testing small value saved data.
        :param uneven_spacing: For unevenly spaced test data.
        """

        # Set precalculated expected interpolation results  using scipy.interpolate.interp1d(kind=linear).
        # This is the result of sampling data on self.xsamples, self.ysamples.
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
        # Set precalculated expected extrapolation results.
        # This is the result of the type of extrapolation on self.xsamples_in_bounds, self.ysamples_in_bounds.
        extrapolation_data = self.setup_extrpolation_type(extrapolator_type)

        # Set the interpolator.
        if uneven_spacing:
            interpolator = Interpolator2DArray(
                self.x_uneven, self.y_uneven, data, 'linear', extrapolator_type, extrapolation_range,
                extrapolation_range
            )

        else:
            interpolator = Interpolator2DArray(
                self.x, self.y, data, 'linear', extrapolator_type, extrapolation_range, extrapolation_range
            )

        return interpolator, interpolation_data, extrapolation_data

    def setup_cubic(self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool,
                    uneven_spacing: bool):
        """
        Sets precalculated values for cubic interpolator.
        Called in every test method that addresses cubic interpolation.

        interpolator will hold Interpolate2DArray object that is being tested.

        :param extrapolator_type: type of extrapolator 'none' or 'linear'.
        :param extrapolation_range: padding around interpolation range where extrapolation is possible.
        :param big_values: For loading and testing big value saved data.
        :param small_values: For loading and testing small value saved data.
        :param uneven_spacing: For unevenly spaced test data.
        """

        # set precalculated expected interpolation results
        # this is the result of sampling data on self.xsamples
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
        # set interpolator
        if uneven_spacing:
            interpolator = Interpolator2DArray(
                self.x_uneven, self.y_uneven, data, 'cubic', extrapolator_type, extrapolation_range, extrapolation_range
            )

        else:
            interpolator = Interpolator2DArray(
                self.x, self.y, data, 'cubic', extrapolator_type, extrapolation_range, extrapolation_range
            )

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
                                    f'x ={self.xsamples_in_bounds[i]} and y={self.ysamples_in_bounds[i]} that should be'
                                    f' outside of the interpolator range of {self.x[0]}<=x<={self.x[-1]} and '
                                    f'{self.y[0]}<=y<={self.y[-1]}'):
                interpolator(self.xsamples_in_bounds[i], self.ysamples_in_bounds[i])

    def test_linear_interpolation_extrapolators(self):
        """
        Testing against linear interpolator objects for interpolation and extrapolation agreement.

        Testing against interp2d linear interpolation with kind=none argument for nearest extrapolation from
        linear interpolation. For linear extrapolation, no equivalent function was found, so the linear extrapolation
        was saved (on 12/07/2021) to be compared to future versions for changes.
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
                            interpolator_str='linear values' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, extrapolator_type=extrapolator_type,
                        interpolator_str='linear values' + uneven_spacing_str
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
                            interpolator_str='linear big values' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, extrapolator_type=extrapolator_type,
                        interpolator_str='linear big values' + uneven_spacing_str
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
                            interpolator_str='linear small values' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, extrapolator_type=extrapolator_type,
                        interpolator_str='linear small values' + uneven_spacing_str
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
        Testing against cubic interpolator objects for interpolation and extrapolation agreement.

        Testing against Cherab cubic interpolators and extrapolators, a numerical inverse in Cherab compared with an
        analytic inverse in the tested interpolators means there is not an agreement to 12 significant figures that the
        data are saved to, but taken to 7 significant figures. An exception for the linear extrapolator is taken because
        linear extrapolation is calculated differently to Cherab, because Cherab duplicates the boundary of the spline
        knot array to get derivatives at the array edge, whereas the tested interpolator object calculates the
        derivative at the edge of the spline knot array as special cases for each edge. The linear extrapolation is
        taken from the current version of interpolators (12/07/2021) and used to test against unexpected changes rather
        than to test consistency in the maths.

        Uneven spacing tests are not tested against Cherab but against a saved version instead.
        """
        # All cubic extrapolators and interpolators are accurate at least to 7 significant figures.
        significant_tolerance = 7
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

                        if extrapolator_type == 'linear':
                            significant_tolerance_extrapolation = None

                        else:
                            significant_tolerance_extrapolation = significant_tolerance

                        self.run_general_extrapolation_tests(
                            interpolator, extrapolation_data, extrapolator_type=extrapolator_type,
                            significant_tolerance=significant_tolerance_extrapolation,
                            interpolator_str='cubic values' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, significant_tolerance=significant_tolerance,
                        extrapolator_type=extrapolator_type, interpolator_str='cubic values' + uneven_spacing_str
                    )

                else:
                    with self.assertRaises(ValueError, msg=f'cubic interpolation and {extrapolator_type} extrapolation'
                                                           f'are not compatible types, yet no ValueError was raised'
                                                           f'setting of types are found in '
                                                           f'permitted_interpolation_combinations'):
                        self.setup_cubic(
                            extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False,
                            uneven_spacing=uneven_spacing
                        )

            # Tests for big values
            for extrapolator_type in id_to_extrapolator.keys():
                if extrapolator_type in permitted_interpolation_combinations['cubic']:
                    interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                        extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False,
                        uneven_spacing=uneven_spacing
                    )

                    if extrapolator_type != 'none':

                        if extrapolator_type == 'linear':
                            significant_tolerance_extrapolation = None

                        else:
                            significant_tolerance_extrapolation = significant_tolerance

                        self.run_general_extrapolation_tests(
                            interpolator, extrapolation_data,
                            extrapolator_type=extrapolator_type,
                            significant_tolerance=significant_tolerance_extrapolation,
                            interpolator_str='cubic big values' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, significant_tolerance=significant_tolerance,
                        extrapolator_type=extrapolator_type,
                        interpolator_str='cubic big values' + uneven_spacing_str
                    )

                else:
                    with self.assertRaises(ValueError, msg=f'cubic interpolation and {extrapolator_type} extrapolation'
                                                           f'are not compatible types, yet no ValueError was raised'
                                                           f'setting of types are found in '
                                                           f'permitted_interpolation_combinations'):
                        self.setup_cubic(
                            extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False,
                            uneven_spacing=uneven_spacing
                        )

            # Tests for small values
            for extrapolator_type in id_to_extrapolator.keys():
                if extrapolator_type in permitted_interpolation_combinations['cubic']:
                    interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                        extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True,
                        uneven_spacing=uneven_spacing
                    )

                    if extrapolator_type != 'none':

                        if extrapolator_type == 'linear':
                            significant_tolerance_extrapolation = None

                        else:
                            significant_tolerance_extrapolation = significant_tolerance

                        self.run_general_extrapolation_tests(
                            interpolator, extrapolation_data,
                            extrapolator_type=extrapolator_type,
                            significant_tolerance=significant_tolerance_extrapolation,
                            interpolator_str='cubic small values' + uneven_spacing_str
                        )

                    self.run_general_interpolation_tests(
                        interpolator, interpolation_data, significant_tolerance=significant_tolerance,
                        extrapolator_type=extrapolator_type,
                        interpolator_str='cubic small values' + uneven_spacing_str
                    )

                else:
                    with self.assertRaises(ValueError, msg=f'cubic interpolation and {extrapolator_type} extrapolation'
                                                           f'are not compatible types, yet no ValueError was raised'
                                                           f'setting of types are found in '
                                                           f'permitted_interpolation_combinations'):
                        self.setup_cubic(
                            extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True,
                            uneven_spacing=uneven_spacing
                        )

    def run_general_extrapolation_tests(
            self, interpolator, extrapolation_data, extrapolator_type='', significant_tolerance=None,
            interpolator_str=''):
        """
        Run general tests for extrapolators.

        Only excluding extrapolator_type 'none', test matching extrapolation inside extrapolation ranges, and raises a
        ValueError outside of the extrapolation ranges.
        """
        # Test extrapolator out of range, there should be an error raised.
        for i in range(len(self.xsamples_out_of_bounds)):
            with self.assertRaises(
                    ValueError, msg=f'No ValueError raised when testing interpolator type {interpolator_str} '
                                    f'extrapolator type {extrapolator_type}, at point x ='
                                    f'{self.xsamples_out_of_bounds[i]} y = {self.ysamples_out_of_bounds[i]} that '
                                    f'should be outside of the interpolator range of {self.x[0]}<=x<={self.x[-1]},  '
                                    f'{self.y[0]}<=y<={self.y[-1]} and also outside of the extrapolation range '
                                    f'{EXTRAPOLATION_RANGE} from these edges.'):
                interpolator(x=self.xsamples_out_of_bounds[i], y=self.ysamples_out_of_bounds[i])

        # Test extrapolation inside extrapolation range matches the predefined values.
        for i in range(len(self.xsamples_in_bounds)):
            if significant_tolerance is None:
                delta_max = np.abs(extrapolation_data[i]/np.power(10., PRECISION - 1))

            else:
                delta_max = np.abs(extrapolation_data[i] * 10**(-significant_tolerance))

            self.assertAlmostEqual(
                interpolator(
                    self.xsamples_in_bounds[i], self.ysamples_in_bounds[i]), extrapolation_data[i],
                delta=delta_max, msg=f'Failed for interpolator {interpolator_str} with extrapolator {extrapolator_type}'
                                     f', attempting to extrapolate at point x ={self.xsamples_in_bounds[i]}, '
                                     f'y ={self.ysamples_in_bounds[i]} that should be outside of the interpolator '
                                     f'range of {self.x[0]}<=x<={self.x[-1]}, {self.y[0]}<=y<={self.y[-1]} and '
                                     f'inside the extrapolation range {EXTRAPOLATION_RANGE} from these edges.'
            )

    def run_general_interpolation_tests(
            self, interpolator, interpolation_data, significant_tolerance=None, extrapolator_type='',
            interpolator_str=''):
        """
        Run general tests for interpolators to match the test data.
        """
        # Test interpolation against xsample
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                if significant_tolerance is None:
                    delta_max = np.abs(interpolation_data[i, j] / np.power(10., PRECISION - 1))

                else:
                    delta_max = np.abs(interpolation_data[i, j] * 10 ** (-significant_tolerance))

                self.assertAlmostEqual(
                    interpolator(self.xsamples[i], self.ysamples[j]), interpolation_data[i, j],
                    delta=delta_max,
                    msg=f'Failed for interpolator {interpolator_str} with extrapolator {extrapolator_type}, attempting '
                        f'to interpolate at point x ={self.xsamples[i]}, y ={self.ysamples[i]} that should be inside of'
                        f' the interpolator range of {self.x[0]}<=x<={self.x[-1]}, {self.y[0]}<=y<={self.y[-1]}.'

                )

    def initialise_tests_on_interpolators(self, x_values, y_values, f_values, problem_str=''):
        """
        Method to create a new interpolator with different x, y, f values to test input failures.
        """
        # Test for all combinations
        for extrapolator_type in id_to_extrapolator.keys():
            for interpolator_type in id_to_interpolator.keys():
                with self.assertRaises(
                        ValueError, msg=f'No ValueError raised when testing interpolator type {interpolator_type} '
                                        f'extrapolator type {extrapolator_type}, trying to initialise a test with '
                                        f'incorrect {problem_str}.'):
                    Interpolator2DArray(
                        x=x_values, y=y_values, f=f_values,
                        interpolation_type=interpolator_type, extrapolation_type=extrapolator_type,
                        extrapolation_range_x=2.0, extrapolation_range_y=2.0
                    )

    def test_incorrect_spline_knots(self):
        """
        Test for bad data being supplied to the interpolators, x, y inputs must be increasing.

        Test x, y monotonically increases, test if x, y spline knots have repeated values.

        """
        # monotonicity x
        x_wrong = np.copy(self.x)
        x_wrong[0] = self.x[1]
        x_wrong[1] = self.x[0]
        self.initialise_tests_on_interpolators(
            x_wrong, self.y, self.reference_loaded_values.data,
            problem_str='monotonicity with the first and second x spline knot the wrong way around'
        )

        # monotonicity y
        y_wrong = np.copy(self.y)
        y_wrong[0] = self.y[1]
        y_wrong[1] = self.y[0]
        self.initialise_tests_on_interpolators(
            self.x, y_wrong, self.reference_loaded_values.data,
            problem_str='monotonicity with the first and second y spline knot the wrong way around'

        )

        # test repeated coordinate x
        x_wrong = np.copy(self.x)
        x_wrong[0] = x_wrong[1]
        self.initialise_tests_on_interpolators(
            x_wrong, self.y, self.reference_loaded_values.data,
            problem_str='the first spline knot is a repeat of the second x spline knot'
        )

        # test repeated coordinate y
        y_wrong = np.copy(self.y)
        y_wrong[0] = y_wrong[1]
        self.initialise_tests_on_interpolators(
            self.x, y_wrong, self.reference_loaded_values.data,
            problem_str='the first spline knot is a repeat of the second y spline knot'

        )

        # mismatch array size between x and data
        x_wrong = np.copy(self.x)
        x_wrong = x_wrong[:-1]
        self.initialise_tests_on_interpolators(
            x_wrong, self.y, self.reference_loaded_values.data, problem_str='the last x spline knot has been removed'
        )

        # mismatch array size between y and data
        y_wrong = np.copy(self.y)
        y_wrong = y_wrong[:-1]
        self.initialise_tests_on_interpolators(
            self.x, y_wrong, self.reference_loaded_values.data, problem_str='the last y spline knot has been removed'
        )

    def test_incorrect_array_length(self):
        """
        Make array inputs have length 1 (too short) in 1 or more dimensions. Then check for a ValueError.
        """

        x = [np.copy(self.x), np.array([np.copy(self.x)[0]])]
        y = [np.copy(self.y), np.array([np.copy(self.y)[0]])]
        f = [
            np.copy(self.reference_loaded_values.data), np.array([[np.copy(self.reference_loaded_values.data)[0, 0]]]),
            np.reshape(np.copy(self.reference_loaded_values.data)[0, :], (1, -1)),
            np.reshape(np.copy(self.reference_loaded_values.data)[:, 0], (-1, 1))
        ]

        incorrect_x = [False, True]
        incorrect_y = [False, True]
        incorrect_fx = [False, True, True, False]
        incorrect_fy = [False, True, False, True]
        for i in range(len(x)):
            if incorrect_x[i]:
                x_str = 'x'

            else:
                x_str = ''

            for j in range(len(y)):
                if incorrect_y[j]:
                    y_str = 'y'

                else:
                    y_str = ''

                for k in range(len(f)):
                    if incorrect_fx[k]:
                        fx_str = 'f in x'

                    else:
                        fx_str = ''

                    if incorrect_fy[k]:
                        fy_str = 'f in y'

                    else:
                        fy_str = ''

                    if not (i == 0 and j == 0 and k == 0):
                        problem_str = f'there is only 1 spline knot in: ({x_str}, {y_str}, {fx_str}, {fy_str})'
                        self.initialise_tests_on_interpolators(x[i], y[j], f[k], problem_str=problem_str)

    def test_incorrect_array_dimension(self):
        """
        Make array inputs have higher dimensions or lower dimensions. Then check for a ValueError.
        """

        x = [np.copy(self.x),
             np.array(np.concatenate((np.copy(self.x)[:, np.newaxis], np.copy(self.x)[:, np.newaxis]), axis=1)),
             np.array(np.copy(self.x)[0])]
        y = [np.copy(self.y),
             np.array(np.concatenate((np.copy(self.y)[:, np.newaxis], np.copy(self.y)[:, np.newaxis]), axis=1)),
             np.array(np.copy(self.y)[0])]
        f = [
            np.copy(self.reference_loaded_values.data), np.array(np.copy(self.reference_loaded_values.data)[0, 0]),
            np.array(np.concatenate((np.copy(self.reference_loaded_values.data)[:, :, np.newaxis],
                                     np.copy(self.reference_loaded_values.data)[:, :, np.newaxis]), axis=2))
        ]
        incorrect_x_short = [False, False, True]
        incorrect_y_short = [False, False, True]
        incorrect_fx_short = [False, True, False]
        incorrect_fy_short = [False, True, False]
        incorrect_x_long = [False, True, False]
        incorrect_y_long = [False, True, False]
        incorrect_fx_long = [False, False, True]
        incorrect_fy_long = [False, False, True]
        for i in range(len(x)):
            if incorrect_x_long[i]:
                x_str_long = 'x'

            else:
                x_str_long = ''

            if incorrect_x_short[i]:
                x_str_short = 'x'

            else:
                x_str_short = ''

            for j in range(len(y)):
                if incorrect_y_long[j]:
                    y_str_long = 'y'

                else:
                    y_str_long = ''

                if incorrect_y_short[j]:
                    y_str_short = 'y'

                else:
                    y_str_short = ''

                for k in range(len(f)):
                    if incorrect_fx_long[k]:
                        fx_str_long = 'f in x'

                    else:
                        fx_str_long = ''

                    if incorrect_fy_long[k]:
                        fy_str_long = 'f in y'

                    else:
                        fy_str_long = ''

                    if incorrect_fx_short[k]:
                        fx_str_short = 'f in x'

                    else:
                        fx_str_short = ''

                    if incorrect_fy_short[k]:
                        fy_str_short = 'f in y'

                    else:
                        fy_str_short = ''

                    if not (i == 0 and j == 0 and k == 0):
                        problem_str = f'there is spline knot array length is too long in : ({x_str_long}, ' \
                                      f'{y_str_long}, {fx_str_long}, {fy_str_long}), too short in : ({x_str_short}, ' \
                                      f'{y_str_short}, {fx_str_short}, {fy_str_short})'
                        self.initialise_tests_on_interpolators(x[i], y[j], f[k], problem_str=problem_str)
