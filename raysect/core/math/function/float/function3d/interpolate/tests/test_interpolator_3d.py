
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
Unit tests for array interpolation (not mesh) from within Interpolate3DArray,
including interaction with internal extrapolators.
"""
import unittest
import numpy as np
from raysect.core.math.function.float.function3d.interpolate.interpolator3darray import Interpolator3DArray, \
    id_to_extrapolator, id_to_interpolator
from raysect.core.math.function.float.function3d.interpolate.tests.scripts.generate_3d_splines import X_LOWER, X_UPPER,\
    NB_XSAMPLES, NB_X, X_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN, PRECISION, Y_LOWER, Y_UPPER, NB_YSAMPLES, NB_Y, \
    Y_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MIN, EXTRAPOLATION_RANGE, large_extrapolation_range, Z_LOWER, Z_UPPER, \
    NB_ZSAMPLES, NB_Z, Z_EXTRAP_DELTA_MAX, Z_EXTRAP_DELTA_MIN, N_EXTRAPOLATION, extrapolation_out_of_bound_points
from raysect.core.math.function.float.function3d.interpolate.tests.data_store.interpolator3d_test_data import \
    TestInterpolatorLoadBigValues, TestInterpolatorLoadNormalValues, TestInterpolatorLoadSmallValues


class TestInterpolators3D(unittest.TestCase):
    """
    Testing class for 3D interpolators and extrapolators.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # data is a precalculated input array for testing. It's the result of applying function f on self.x,
        # self.y, self.z to create data = f(self.x,self.y,self.z), where self.x, self.y ,self.z are linearly
        # spaced between X_LOWER and X_UPPER ...

        #: x, y and z values used to obtain data.
        x_in = np.linspace(X_LOWER, X_UPPER, NB_X)
        y_in = np.linspace(Y_LOWER, Y_UPPER, NB_Y)
        z_in = np.linspace(Z_LOWER, Z_UPPER, NB_Z)
        cls.x = x_in
        cls.y = y_in
        cls.z = z_in

        cls.reference_loaded_values = TestInterpolatorLoadNormalValues()
        cls.reference_loaded_big_values = TestInterpolatorLoadBigValues()
        cls.reference_loaded_small_values = TestInterpolatorLoadSmallValues()

        #: x, y, z values on which interpolation_data was sampled on.
        cls.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
        cls.ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)
        cls.zsamples = np.linspace(Z_LOWER, Z_UPPER, NB_ZSAMPLES)

        # Extrapolation x, y and z values.
        cls.xsamples_out_of_bounds, cls.ysamples_out_of_bounds, cls.zsamples_out_of_bounds = extrapolation_out_of_bound_points(X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, Z_LOWER, Z_UPPER, X_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MAX, Z_EXTRAP_DELTA_MAX, EXTRAPOLATION_RANGE)
        cls.xsamples_in_bounds, cls.ysamples_in_bounds, cls.zsamples_in_bounds = large_extrapolation_range(cls.xsamples, cls.ysamples, cls.zsamples, EXTRAPOLATION_RANGE, N_EXTRAPOLATION)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Remove the larger classes holding load values
        """
        try:
            del cls.reference_loaded_values
        except AttributeError:
            pass
        try:
            del cls.reference_loaded_big_values
        except AttributeError:
            pass
        try:
            del cls.reference_loaded_small_values
        except AttributeError:
            pass

    def setup_linear(self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool):
        """
        Sets precalculated values for linear interpolator.
        Called in every test method that addresses linear interpolation.

        interpolator will hold Interpolate3DArray object that is being tested.

        :param extrapolator_type: type of extrapolator 'none' or 'linear'.
        :param extrapolation_range: padding around interpolation range where extrapolation is possible.
        :param big_values: For loading and testing big value saved data.
        :param small_values: For loading and testing small value saved data.
        """

        if big_values:
            self.value_storage_obj = self.reference_loaded_big_values
        elif small_values:
            self.value_storage_obj = self.reference_loaded_small_values
        else:
            self.value_storage_obj = self.reference_loaded_values

        self.value_storage_obj.setup_linear()
        data = self.value_storage_obj.data
        interpolation_data = self.value_storage_obj.precalc_interpolation
        # set precalculated expected extrapolation results
        # this is the result of the type of extrapolation on self.xsamples_extrap
        extrapolation_data = self.setup_extrpolation_type(extrapolator_type)

        # set interpolator
        interpolator = Interpolator3DArray(
            self.x, self.y, self.z, data, 'linear', extrapolator_type, extrapolation_range, extrapolation_range,
            extrapolation_range
        )
        return interpolator, interpolation_data, extrapolation_data

    def setup_cubic(self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool):
        """
        Sets precalculated values for cubic interpolator.
        Called in every test method that addresses cubic interpolation.

        interpolator will hold Interpolate3DArray object that is being tested.

        :param extrapolator_type: type of extrapolator 'none' or 'linear'.
        :param extrapolation_range: padding around interpolation range where extrapolation is possible.
        :param big_values: For loading and testing big value saved data.
        :param small_values: For loading and testing small value saved data.
        """

        # Set precalculated expected interpolation results.
        # This is the result of sampling data on self.xsamples, self.ysamples, self.zsamples.
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
        # Set the interpolator.
        interpolator = Interpolator3DArray(
            self.x, self.y, self.z, data, 'cubic', extrapolator_type, extrapolation_range, extrapolation_range,
            extrapolation_range
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
            raise ValueError(
                f'Extrapolation type {extrapolator_type} not found or no test. options are {id_to_extrapolator.keys()}'
            )
        return extrapolation_data

    def test_extrapolation_none(self):
        """
        Testing that extrapolator_type 'none' returns a ValueError rather than data when attempting to extrapolate
        outside its extrapolation range.
        """
        interpolator, _, _ = self.setup_linear('none', EXTRAPOLATION_RANGE, big_values=False, small_values=False)
        for i in range(len(self.xsamples_in_bounds)):
            self.assertRaises(
                ValueError, interpolator, x=self.xsamples_in_bounds[i], y=self.ysamples_in_bounds[i],
                z=self.zsamples_in_bounds[i]
            )

    def test_linear_interpolation_extrapolators(self):
        """
        Testing against linear interpolator objects for interpolation and extrapolation agreement.

        Testing against Cherab linear interpolators with nearest neighbour extrapolators for agreement. For linear
        extrapolation, the derivatives at the edges of the spline knots are calculated differently to Cherab, so
        the linear extrapolation is saved (on 12/07/2021) to be compared to future versions for changes.
        """
        no_test_for_extrapolator = []
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_linear(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False
            )
            if extrapolator_type != 'none':
                if extrapolator_type not in no_test_for_extrapolator:
                    self.run_general_extrapolation_tests(interpolator, extrapolation_data, extrapolator_type=extrapolator_type, interpolator_str='linear')
            self.run_general_interpolation_tests(interpolator, interpolation_data, interpolator_str='linear')

        # Tests for big values
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_linear(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False
            )
            if extrapolator_type != 'none':
                if extrapolator_type not in no_test_for_extrapolator:
                    self.run_general_extrapolation_tests(interpolator, extrapolation_data, extrapolator_type=extrapolator_type, interpolator_str='linear')

            self.run_general_interpolation_tests(interpolator, interpolation_data, interpolator_str='linear')

        # Tests for small values
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_linear(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True
            )
            if extrapolator_type != 'none':
                if extrapolator_type not in no_test_for_extrapolator:
                    self.run_general_extrapolation_tests(interpolator, extrapolation_data, extrapolator_type=extrapolator_type, interpolator_str='linear')

            self.run_general_interpolation_tests(interpolator, interpolation_data, interpolator_str='linear')

    def test_cubic_interpolation_extrapolators(self):
        """
        Testing against cubic interpolator objects for interpolation and extrapolation agreement.

        Testing against Cherab cubic interpolators and extrapolators, a numerical inverse in Cherab compared with an
        analytic inverse in the tested interpolators means there is not an agreement to 12 significant figures that the
        data are saved to, but taken to 4 significant figures. An exception for the linear extrapolator is made because
        linear extrapolation is calculated differently to Cherab, because Cherab duplicates the boundary of the spline
        knot array to get derivatives at the array edge, whereas the tested interpolator object calculates the
        derivative at the edge of the spline knot array as special cases for each edge. The linear extrapolation is
        taken from the current version of interpolators (12/07/2021) and used to test against unexpected changes rather
        than to test consistency in the maths.
        """

        # All cubic extrapolators and interpolators are accurate at least to 4 significant figures.
        significant_tolerance = 4

        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False
            )
            if extrapolator_type == 'linear':
                significant_tolerance_extrapolation = None
            else:
                significant_tolerance_extrapolation = significant_tolerance

            if extrapolator_type != 'none':
                self.run_general_extrapolation_tests(
                    interpolator, extrapolation_data,
                    extrapolator_type=extrapolator_type, significant_tolerance=significant_tolerance_extrapolation, interpolator_str='cubic values'
                )
            self.run_general_interpolation_tests(interpolator, interpolation_data, significant_tolerance=significant_tolerance, interpolator_str='cubic values')

        # Tests for big values
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False
            )
            if extrapolator_type == 'linear':
                significant_tolerance_extrapolation = None
            else:
                significant_tolerance_extrapolation = significant_tolerance

            if extrapolator_type != 'none':
                self.run_general_extrapolation_tests(
                    interpolator, extrapolation_data,
                    extrapolator_type=extrapolator_type, significant_tolerance=significant_tolerance_extrapolation, interpolator_str='cubic big values'
                )
            self.run_general_interpolation_tests(interpolator, interpolation_data, significant_tolerance=significant_tolerance, interpolator_str='cubic big values')

        # Tests for small values
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True
            )
            if extrapolator_type == 'linear':
                significant_tolerance_extrapolation = None
            else:
                significant_tolerance_extrapolation = significant_tolerance

            if extrapolator_type != 'none':
                self.run_general_extrapolation_tests(
                    interpolator, extrapolation_data,
                    extrapolator_type=extrapolator_type, significant_tolerance=significant_tolerance_extrapolation, interpolator_str='cubic small values'
                )

            self.run_general_interpolation_tests(interpolator, interpolation_data, significant_tolerance=significant_tolerance, interpolator_str='cubic small values')

    def run_general_extrapolation_tests(self, interpolator, extrapolation_data, extrapolator_type='', significant_tolerance=None, interpolator_str=''):
        """
        Run general tests for extrapolators.

        Only excluding extrapolator_type 'none', test matching extrapolation inside extrapolation ranges, and raises a
        ValueError outside of the extrapolation ranges.
        """
        # Test extrapolator out of range, there should be an error raised.
        for i in range(len(self.xsamples_out_of_bounds)):
            self.assertRaises(
                ValueError, interpolator, x=self.xsamples_out_of_bounds[i], y=self.ysamples_out_of_bounds[i],
                z=self.zsamples_out_of_bounds[i]
            )

        # Test extrapolation inside extrapolation range matches the predefined values
        for i in range(len(self.xsamples_in_bounds)):
            if significant_tolerance is None:
                delta_max = np.abs(extrapolation_data[i]/np.power(10., PRECISION - 1))
            else:
                delta_max = np.abs(extrapolation_data[i] * 10**(-significant_tolerance))
            self.assertAlmostEqual(
                interpolator(
                    self.xsamples_in_bounds[i], self.ysamples_in_bounds[i], self.zsamples_in_bounds[i]),
                extrapolation_data[i],
                delta=delta_max, msg=f'Failed extrapolation attempt for interpolator {interpolator_str} and extrapolator {extrapolator_type}'
                                     f'{self.xsamples_in_bounds[i]}, {self.ysamples_in_bounds[i]} and '
                                     f'{self.zsamples_in_bounds[i]}'
            )

    def run_general_interpolation_tests(self, interpolator, interpolation_data, significant_tolerance=None, interpolator_str=''):
        """
        Run general tests for interpolators to match the test data.
        """
        # Test interpolation against xsample
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                for k in range(len(self.zsamples)):
                    if significant_tolerance is None:
                        delta_max = np.abs(interpolation_data[i, j, k] / np.power(10., PRECISION - 1))
                    else:
                        delta_max = np.abs(interpolation_data[i, j, k] * 10**(-significant_tolerance))
                    self.assertAlmostEqual(
                        interpolator(self.xsamples[i], self.ysamples[j], self.zsamples[k]),
                        interpolation_data[i, j, k], delta=delta_max
                    )

    def initialise_tests_on_interpolators(self, x_values, y_values, z_values, f_values):
        """
        Method to create a new interpolator with different x, y, z, f values to test input failures.
        """
        # Test for all combinations
        for extrapolator_type in id_to_extrapolator.keys():
            for interpolator_type in id_to_interpolator.keys():
                self.assertRaises(
                    ValueError, Interpolator3DArray, x=x_values, y=y_values, z=z_values, f=f_values,
                    interpolation_type=interpolator_type, extrapolation_type=extrapolator_type,
                    extrapolation_range_x=2.0, extrapolation_range_y=2.0, extrapolation_range_z=2.0
                )

    def test_initialisation_errors(self):
        """
        Test for bad data being supplied to the interpolators

        Test x, y, z monotonically increases, test if an x, y, z is repeated, test if arrays are different lengths.

        """
        # monotonicity x
        x_wrong = np.copy(self.x)
        x_wrong[0] = self.x[1]
        x_wrong[1] = self.x[0]
        self.initialise_tests_on_interpolators(x_wrong, self.y, self.z, self.reference_loaded_values.data)

        # monotonicity y
        y_wrong = np.copy(self.y)
        y_wrong[0] = self.y[1]
        y_wrong[1] = self.y[0]
        self.initialise_tests_on_interpolators(self.x, y_wrong, self.z, self.reference_loaded_values.data)

        # monotonicity z
        z_wrong = np.copy(self.z)
        z_wrong[0] = self.z[1]
        z_wrong[1] = self.z[0]
        self.initialise_tests_on_interpolators(self.x, self.y, z_wrong, self.reference_loaded_values.data)

        # test repeated coordinate x
        x_wrong = np.copy(self.x)
        x_wrong[0] = x_wrong[1]
        self.initialise_tests_on_interpolators(x_wrong, self.y, self.z, self.reference_loaded_values.data)

        # test repeated coordinate y
        y_wrong = np.copy(self.y)
        y_wrong[0] = y_wrong[1]
        self.initialise_tests_on_interpolators(self.x, y_wrong, self.z, self.reference_loaded_values.data)

        # test repeated coordinate z
        z_wrong = np.copy(self.z)
        z_wrong[0] = z_wrong[1]
        self.initialise_tests_on_interpolators(self.x, self.y, z_wrong, self.reference_loaded_values.data)

        # mismatch array size between x and data
        x_wrong = np.copy(self.x)
        x_wrong = x_wrong[:-1]
        self.initialise_tests_on_interpolators(x_wrong, self.y, self.z, self.reference_loaded_values.data)

        # mismatch array size between y and data
        y_wrong = np.copy(self.y)
        y_wrong = y_wrong[:-1]
        self.initialise_tests_on_interpolators(self.x, y_wrong, self.z, self.reference_loaded_values.data)

        # mismatch array size between z and data
        z_wrong = np.copy(self.z)
        z_wrong = z_wrong[:-1]
        self.initialise_tests_on_interpolators(self.x, self.y, z_wrong, self.reference_loaded_values.data)

        # Test array length 1, Arrays are too short.
        self.run_incorrect_array_length_combination()

        # Incorrect dimensional data supplied.
        self.run_incorrect_array_dimension_combination()

    def run_incorrect_array_length_combination(self):
        """
        Make array inputs have length 1 (too short) in 1 or more dimensions. Then check for a ValueError.
        """

        x = [np.copy(self.x), np.copy(self.x)[0]]
        y = [np.copy(self.y), np.copy(self.y)[0]]
        z = [np.copy(self.z), np.copy(self.z)[0]]
        f = [
            np.copy(self.reference_loaded_values.data), np.copy(self.reference_loaded_values.data)[0, 0, 0],
            np.copy(self.reference_loaded_values.data)[0, 0, :], np.copy(self.reference_loaded_values.data)[0, :, 0],
            np.copy(self.reference_loaded_values.data)[:, 0, 0], np.copy(self.reference_loaded_values.data)[0, :, :],
            np.copy(self.reference_loaded_values.data)[:, 0, :], np.copy(self.reference_loaded_values.data)[:, :, 0]
        ]
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    for i2 in range(len(f)):
                        if not (i == 0 and j == 0 and k == 0 and i2 == 0):
                            self.initialise_tests_on_interpolators(x[i], y[j], z[k], f[i2])

    def run_incorrect_array_dimension_combination(self):
        """
        Make array inputs have higher dimensions or lower dimensions. Then check for a ValueError.
        """

        x = [np.copy(self.x),
             np.array(np.concatenate((np.copy(self.x)[:, np.newaxis], np.copy(self.x)[:, np.newaxis]), axis=1)),
             np.array(np.copy(self.x)[0])]
        y = [np.copy(self.y),
             np.array(np.concatenate((np.copy(self.y)[:, np.newaxis], np.copy(self.y)[:, np.newaxis]), axis=1)),
             np.array(np.copy(self.y)[0])]
        z = [np.copy(self.z),
             np.array(np.concatenate((np.copy(self.z)[:, np.newaxis], np.copy(self.z)[:, np.newaxis]), axis=1)),
             np.array(np.copy(self.z)[0])]
        f = [
            np.copy(self.reference_loaded_values.data), np.array(np.copy(self.reference_loaded_values.data)[0, 0, 0]),
            np.array(np.concatenate((np.copy(self.reference_loaded_values.data)[:, :, :, np.newaxis],
                                     np.copy(self.reference_loaded_values.data)[:, :, :, np.newaxis]), axis=3))
        ]

        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    for i2 in range(len(f)):
                        if not (i == 0 and j == 0 and k == 0 and i2 == 0):
                            self.initialise_tests_on_interpolators(x[i], y[j], z[k], f[i2])
