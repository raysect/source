
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

import numpy as np
from raysect.core.math.function.float.function2d.interpolate.interpolator2darray import Interpolator2DArray
from scipy.interpolate import griddata, interp2d, RectBivariateSpline, CloughTocher2DInterpolator
from raysect.core.math.function.float.function2d.interpolate.tests.data_store.interpolator2d_test_data import \
    TestInterpolatorLoadBigValues, TestInterpolatorLoadNormalValues, TestInterpolatorLoadSmallValues, \
    TestInterpolatorLoadBigValuesUneven, TestInterpolatorLoadNormalValuesUneven, TestInterpolatorLoadSmallValuesUneven
import scipy

X_LOWER = -1.0
X_UPPER = 1.0
Y_LOWER = -1.0
Y_UPPER = 1.0
Y_EXTRAP_DELTA_MAX = 0.08
Y_EXTRAP_DELTA_MIN = 0.04
X_EXTRAP_DELTA_MAX = 0.08
X_EXTRAP_DELTA_MIN = 0.04

NB_X = 10
NB_Y = 10
NB_XSAMPLES = 13
NB_YSAMPLES = 13

N_EXTRAPOLATION = 3

EXTRAPOLATION_RANGE = 2.0

PRECISION = 12

BIG_VALUE_FACTOR = 20.
SMALL_VALUE_FACTOR = -20.


# Force scientific format to get the right number of significant figures
np.set_printoptions(30000, linewidth=100, formatter={'float': lambda x_str: format(x_str, '.'+str(PRECISION)+'E')})


def function_to_spline(x_input, y_input, factor_in):
    t = np.pi * np.sqrt((x_input ** 2 + y_input ** 2))
    return factor_in*np.sinc(t)


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


def extrapolation_out_of_bound_points(x_lower, x_upper, y_lower, y_upper, x_extrap_delta_max, y_extrap_delta_max, extrapolation_range):
    xsamples_extrap_out_of_bounds_options = np.array(
        [x_lower - extrapolation_range - x_extrap_delta_max, (x_lower + x_upper) / 2., x_upper + extrapolation_range + x_extrap_delta_max])

    ysamples_extrap_out_of_bounds_options = np.array(
        [y_lower - extrapolation_range - y_extrap_delta_max, (y_lower + y_upper) / 2., y_upper + extrapolation_range + y_extrap_delta_max])

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


if __name__ == '__main__':
    # Calculate for big values, small values, or normal values
    big_values = False
    small_values = True

    uneven_spacing = True
    use_saved_datastore_spline_knots = True

    print('Using scipy version', scipy.__version__)

    # Find the function values to be used
    if big_values:
        factor = np.power(10., BIG_VALUE_FACTOR)
    elif small_values:
        factor = np.power(10., SMALL_VALUE_FACTOR)
    else:
        factor = 1.

    if uneven_spacing:
        x_in = uneven_linspace(X_LOWER, X_UPPER, NB_X, offset_fraction=1./3.)
        y_in = uneven_linspace(Y_LOWER, Y_UPPER, NB_Y, offset_fraction=1./3.)
    else:
        x_in = np.linspace(X_LOWER, X_UPPER, NB_X)
        y_in = np.linspace(Y_LOWER, Y_UPPER, NB_Y)
    x_in_full, y_in_full = np.meshgrid(x_in, y_in)
    f_in = function_to_spline(x_in_full, y_in_full, factor)

    if use_saved_datastore_spline_knots:
        if uneven_spacing:
            if big_values:
                reference_loaded_values = TestInterpolatorLoadBigValuesUneven()
            elif small_values:
                reference_loaded_values = TestInterpolatorLoadSmallValuesUneven()
            else:
                reference_loaded_values = TestInterpolatorLoadNormalValuesUneven()
        else:
            if big_values:
                reference_loaded_values = TestInterpolatorLoadBigValues()
            elif small_values:
                reference_loaded_values = TestInterpolatorLoadSmallValues()
            else:
                reference_loaded_values = TestInterpolatorLoadNormalValues()
        f_in = reference_loaded_values.data

    print('Save this to self.data in test_interpolator:\n', repr(f_in))

    # Make the sampled points between spline knots and find the precalc_interpolation for test_interpolator.setup_cubic
    xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
    ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)

    # Make grid
    xsamples_in_full, ysamples_in_full = np.meshgrid(xsamples, ysamples)

    collapsed_xsamples_in_full = np.reshape(xsamples_in_full, -1)
    collapsed_ysamples_in_full = np.reshape(ysamples_in_full, -1)

    # # Extrapolation x and y values
    # xsamples_out_of_bounds, ysamples_out_of_bounds, xsamples_in_bounds,  ysamples_in_bounds = \
    #     get_extrapolation_input_values(
    #         X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, X_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN,
    #         Y_EXTRAP_DELTA_MIN
    #     )

    linear_2d = interp2d(x_in, y_in, f_in, kind='linear')
    f_linear = linear_2d(xsamples, ysamples)
    print('Linear spline at xsamples, ysamples created using. interp2d(kind=linear)',
          'Save this to self.precalc_interpolation in test_interpolator in setup_linear:\n', repr(f_linear))

    interpolator2D = Interpolator2DArray(x_in, y_in, f_in, 'linear', 'nearest', extrapolation_range_x=2.0,
                                         extrapolation_range_y=2.0)
    # f_extrap_nearest_2 = np.zeros((len(xsamples_in_bounds),))
    # for i in range(len(xsamples_in_bounds)):
    #     f_extrap_nearest_2[i] = interpolator2D(xsamples_in_bounds[i], ysamples_in_bounds[i])

    # interp2d - Runs dfitpack.regrid_smth (a fortran code in scipy), if on a rectangular grid.
    cubic_2d = interp2d(x_in, y_in, f_in, kind='cubic')
    f_cubic = cubic_2d(xsamples, ysamples)

    # RectBivariateSpline - Runs from dfitpack.regrid_smth (a fortran code in scipy).
    cubic_2da = RectBivariateSpline(x_in, y_in, f_in, kx=3, ky=3)
    f_cubica = cubic_2da(xsamples, ysamples)

    # CloughTocher2DInterpolator - an iterative piecewise interpolator method.
    x_flat = np.reshape(x_in_full, -1)
    y_flat = np.reshape(y_in_full, -1)
    f_inflat = np.reshape(f_in, -1)
    xy_flat = np.concatenate((x_flat[:, np.newaxis], y_flat[:, np.newaxis]), axis=1)
    interp_clough_tocher = CloughTocher2DInterpolator(xy_flat, f_inflat)

    # griddata - a wrapper for CloughTocher2DInterpolator
    grid_z = griddata(xy_flat, f_inflat, (xsamples_in_full, ysamples_in_full), method='cubic')

    # It is not possible to get exactly the same as other methods for cubic interpolation.
    # The method of getting the coefficients is slightly different. Using a version to check for changes instead.
    interpolator2D_cubic_nearest = Interpolator2DArray(
        x_in, y_in, f_in, 'cubic', 'nearest', extrapolation_range_x=2.0, extrapolation_range_y=2.0
    )
    interpolator2D_cubic_linear = Interpolator2DArray(
        x_in, y_in, f_in, 'cubic', 'linear', extrapolation_range_x=2.0, extrapolation_range_y=2.0
    )
    interpolator2D_linear_linear = Interpolator2DArray(
        x_in, y_in, f_in, 'linear', 'linear', extrapolation_range_x=2.0, extrapolation_range_y=2.0
    )

    f_out = np.zeros((len(xsamples), len(ysamples)))
    for i in range(len(xsamples)):
        for j in range(len(ysamples)):
            f_out[i, j] = interpolator2D_cubic_nearest(xsamples[i], ysamples[j])

    xsamples_extrapolation, ysamples_extrapolation = large_extrapolation_range(xsamples, ysamples, EXTRAPOLATION_RANGE, N_EXTRAPOLATION)

    f_extrap_cubic_nearest = np.zeros((len(xsamples_extrapolation),))
    f_extrap_cubic_linear = np.zeros((len(xsamples_extrapolation),))
    f_extrap_linear_linear = np.zeros((len(xsamples_extrapolation),))
    for i in range(len(xsamples_extrapolation)):
        f_extrap_cubic_nearest[i] = interpolator2D_cubic_nearest(xsamples_extrapolation[i], ysamples_extrapolation[i])
        f_extrap_cubic_linear[i] = interpolator2D_cubic_linear(xsamples_extrapolation[i], ysamples_extrapolation[i])
        f_extrap_linear_linear[i] = interpolator2D_linear_linear(xsamples_extrapolation[i], ysamples_extrapolation[i])
    print('PUT THIS IN CHERAB - Output of nearest neighbour extrapolation from a Cubic spline at xsamples_in_bounds, '
          'ysamples_in_bounds created using the Interpolator2DArray on 05/07/2021. Save this to '
          'self.precalc_extrapolation_nearest in test_interpolator in setup_cubic:\n', repr(f_extrap_cubic_nearest))

    print('Output of linear extrapolation from a Cubic spline at xsamples_in_bounds, ysamples_in_bounds created using '
          'the Interpolator2DArray on 05/07/2021. Save this to self.precalc_extrapolation_linear in test_interpolator '
          'in setup_cubic:\n', repr(f_extrap_cubic_linear))

    print('Output of linear extrapolation from a bilinear spline at xsamples_in_bounds, ysamples_in_bounds created '
          'using the Interpolator2DArray on 05/07/2021. Save this to self.precalc_extrapolation_linear in '
          'test_interpolator in setup_cubic:\n', repr(f_extrap_linear_linear))

    print('Cubic spline at xsamples, ysamples created using the Interpolator2DArray on 05/07/2021',
          'Save this to self.precalc_interpolation in test_interpolator in setup_cubic:\n', repr(f_out))

    linear_2d_nearest_neighbour = interp2d(x_in, y_in, f_in, kind='linear', fill_value=None)
    f_extrap_nearest = np.zeros((len(xsamples_extrapolation),))
    for i in range(len(xsamples_extrapolation)):
        f_extrap_nearest[i] = linear_2d_nearest_neighbour(xsamples_extrapolation[i], ysamples_extrapolation[i])

    print('Output of nearest neighbour extrapolation from linear interpolator at the start and end spline knots ',
          'Save this to self.precalc_extrapolation_nearest in test_interpolator in setup_linear:\n',
          repr(f_extrap_nearest))


    check_plot = True
    if check_plot:
        xsamples_lower_and_upper = np.linspace(X_LOWER-0.1*(X_UPPER-X_LOWER), X_UPPER+0.1*(X_UPPER-X_LOWER), 50)
        ysamples_lower_and_upper = np.linspace(Y_LOWER-0.1*(Y_UPPER-Y_LOWER), Y_UPPER+0.1*(Y_UPPER-Y_LOWER), 50)
        xsamples_lower_and_upper_full, ysamples_lower_and_upper_full = np.meshgrid(
            xsamples_lower_and_upper, ysamples_lower_and_upper
        )

        import matplotlib.pyplot as plt
        from matplotlib import cm
        fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
        surf = ax[0].plot_surface(x_in_full, y_in_full, f_in, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        main_plots_on = True
        interpolator2D = Interpolator2DArray(
            x_in, y_in, f_in, 'linear', 'nearest', extrapolation_range_x=2.0, extrapolation_range_y=2.0
        )

        if main_plots_on:
            f_out = np.zeros((len(xsamples), len(ysamples)))
            for i in range(len(xsamples)):
                for j in range(len(ysamples)):
                    f_out[i, j] = interpolator2D(xsamples[i], ysamples[j])
            f_out_extrap = np.zeros((len(xsamples_extrapolation), ))
            for i in range(len(xsamples_extrapolation)):
                f_out_extrap[i] = interpolator2D(xsamples_extrapolation[i], ysamples_extrapolation[i])

            f_out_lower_and_upper = np.zeros((len(xsamples_lower_and_upper), len(ysamples_lower_and_upper)))
            for i in range(len(xsamples_lower_and_upper)):
                for j in range(len(ysamples_lower_and_upper)):
                    f_out_lower_and_upper[i, j] = interpolator2D(
                        xsamples_lower_and_upper[i], ysamples_lower_and_upper[j]
                    )
            # ax[0].scatter(xsamples_in_full, ysamples_in_full, f_out, color='r')
            ax[0].scatter(xsamples_extrapolation, ysamples_extrapolation, f_out_extrap, color='g')
            # ax[0].scatter(xsamples_in_bounds, ysamples_in_bounds, f_extrap_nearest, color='m')
            # ax[1].scatter(
            # collapsed_xsamples_in_full, collapsed_ysamples_in_full, cubic_2d(xsamples, ysamples), color='m'
            # )
            f_true_points = function_to_spline(xsamples_in_full, ysamples_in_full, factor)
            ax[1].scatter(collapsed_xsamples_in_full, collapsed_ysamples_in_full, f_true_points, color='g')
            # ax[1].scatter(
            # collapsed_xsamples_in_full, collapsed_ysamples_in_full, interp_clough_tocher(xsamples_in_full,
            # ysamples_in_full), color='b')
            # ax[1].scatter(collapsed_xsamples_in_full, collapsed_ysamples_in_full, grid_z, color='k')
            # ax[1].scatter(collapsed_xsamples_in_full, collapsed_ysamples_in_full, f_cubica, color='r')
            surf = ax[1].plot_surface(
                xsamples_in_full, ysamples_in_full, f_out, cmap=cm.coolwarm, linewidth=0, antialiased=False
            )

            surf = ax[2].plot_surface(
                xsamples_lower_and_upper_full, ysamples_lower_and_upper_full, f_out_lower_and_upper, cmap=cm.coolwarm,
                linewidth=0, antialiased=False
            )
            ax[0].set_title('Spline knots')
            ax[1].set_title('Interpolated points for testing')
            ax[2].set_title('Interpolated points for detailed view')
        plt.show()
