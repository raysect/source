
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
from raysect.core.math.function.float.function2d.interpolate.interpolator2dgrid import Interpolator2DGrid
from scipy.interpolate import griddata, interp2d, RectBivariateSpline
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

EXTRAPOLATION_RANGE = 0.06

PRECISION = 12

BIG_VALUE_FACTOR = 20.
SMALL_VALUE_FACTOR = -20.


# Force scientific format to get the right number of significant figures
np.set_printoptions(30000, linewidth=100, formatter={'float': lambda x_str: format(x_str, '.'+str(PRECISION)+'E')})


def function_to_spline(x_input, y_input, factor):
    t = np.pi * np.sqrt((x_input ** 2 + y_input ** 2))
    return factor*np.sinc(t)


def get_extrapolation_input_values(
        x_lower, x_upper, y_lower, y_upper, x_extrap_delta_max, y_extrap_delta_max, x_extrap_delta_min,
        y_extrap_delta_min):
    xsamples_extrap_out_of_bounds_options = np.array(
        [x_lower - x_extrap_delta_max, (x_lower + x_upper) / 2., x_upper + x_extrap_delta_max])
    ysamples_extrap_out_of_bounds_options = np.array(
        [y_lower - y_extrap_delta_max, (y_lower + y_upper) / 2., y_upper + y_extrap_delta_max])
    xsamples_extrap_in_bounds_options = np.array(
        [x_lower - x_extrap_delta_min, (x_lower + x_upper) / 2., x_upper + x_extrap_delta_min])
    ysamples_extrap_in_bounds_options = np.array(
        [y_lower - y_extrap_delta_min, (y_lower + y_upper) / 2., y_upper + y_extrap_delta_min])
    xsamples_extrap_out_of_bounds = []
    ysamples_extrap_out_of_bounds = []
    xsamples_extrap_in_bounds = []
    ysamples_extrap_in_bounds = []
    edge_indicies = [0, len(xsamples_extrap_out_of_bounds_options) - 1]
    for i_x in range(len(xsamples_extrap_out_of_bounds_options)):
        for j_y in range(len(xsamples_extrap_out_of_bounds_options)):
            if not (i_x not in edge_indicies and j_y not in edge_indicies):
                xsamples_extrap_out_of_bounds.append(xsamples_extrap_out_of_bounds_options[i_x])
                ysamples_extrap_out_of_bounds.append(ysamples_extrap_out_of_bounds_options[j_y])
                xsamples_extrap_in_bounds.append(xsamples_extrap_in_bounds_options[i_x])
                ysamples_extrap_in_bounds.append(ysamples_extrap_in_bounds_options[j_y])
    return \
        np.array(xsamples_extrap_out_of_bounds), np.array(ysamples_extrap_out_of_bounds), \
        np.array(xsamples_extrap_in_bounds), np.array(ysamples_extrap_in_bounds)

#
# def get_nearest_neighbour_output_values(factor_in, x_input, y_input):
#     f_nearest_extrap_in_bounds = []
#     x_in_use = [x_input[0], (x_input[-1] + x_input[0])/2., x_in[-1]]
#     y_in_use = [y_input[0], (y_input[-1] + y_input[0])/2., y_in[-1]]
#     edge_indicies = [0, len(x_in_use) - 1]
#
#     for i_x in range(len(x_in_use)):
#         for j_y in range(len(y_in_use)):
#             if not (i_x not in edge_indicies and j_y not in edge_indicies):
#                 f_nearest_extrap_in_bounds.append(list(function_to_spline(x_input, y_input, factor_in)))
#     return np.array(f_nearest_extrap_in_bounds)


if __name__ == '__main__':
    # Calculate for big values, small values, or normal values
    big_values = False
    small_values = False

    print('Using scipy version', scipy.__version__)

    # Find the function values to be used
    if big_values:
        factor = np.power(10., BIG_VALUE_FACTOR)
    elif small_values:
        factor = np.power(10., SMALL_VALUE_FACTOR)
    else:
        factor = 1.

    x_in = np.linspace(X_LOWER, X_UPPER, NB_X)
    y_in = np.linspace(Y_LOWER, Y_UPPER, NB_Y)
    x_in_full, y_in_full = np.meshgrid(x_in, y_in)
    f_in = function_to_spline(x_in_full, y_in_full, factor)

    print('Save this to self.data in test_interpolator:\n', repr(f_in))

    # Make the sampled points between spline knots and find the precalc_interpolation used in test_interpolator.setup_cubic
    xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
    ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)

    # Make grid
    xsamples_in_full, ysamples_in_full = np.meshgrid(xsamples, ysamples)

    collapsed_xsamples_in_full = np.reshape(xsamples_in_full, -1)
    collapsed_ysamples_in_full = np.reshape(ysamples_in_full, -1)

    precalc_interpolation_function_vals = function_to_spline(xsamples_in_full, ysamples_in_full, factor)

    # print('Save this to self.precalc_function in test_interpolator:\n', repr(precalc_interpolation_function_vals))

    # Extrapolation x and y values
    xsamples_out_of_bounds, ysamples_out_of_bounds, xsamples_in_bounds,  ysamples_in_bounds = \
        get_extrapolation_input_values(
            X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, X_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN,
            Y_EXTRAP_DELTA_MIN
        )
    print(xsamples_out_of_bounds, 'xsamples_out_of_bounds')
    print(ysamples_out_of_bounds, 'ysamples_out_of_bounds')
    print(xsamples_in_bounds, 'xsamples_in_bounds')
    print(ysamples_in_bounds, 'ysamples_in_bounds')

    # xsamples_extrap = np.array([
    #     X_LOWER - X_EXTRAP_DELTA_MAX, X_LOWER - X_EXTRAP_DELTA_MIN, X_UPPER + X_EXTRAP_DELTA_MIN,
    #     X_UPPER + X_EXTRAP_DELTA_MAX], dtype=np.float64,
    # )

    linear_2d = interp2d(x_in, y_in, f_in, kind='linear')
    f_linear = linear_2d(xsamples, ysamples)
    print('Linear spline at xsamples, ysamples created using. interp2d(kind=linear)',
          'Save this to self.precalc_interpolation in test_interpolator in setup_linear:\n', repr(f_linear))

    linear_2d_nearest_neighbour = interp2d(x_in, y_in, f_in, kind='linear', fill_value=None)
    f_extrap_nearest = np.zeros((len(xsamples_in_bounds),))
    for i in range(len(xsamples_in_bounds)):
        f_extrap_nearest[i] = linear_2d_nearest_neighbour(xsamples_in_bounds[i], ysamples_in_bounds[i])

    print('Output of nearest neighbour extrapolation from the start and end spline knots ',
          'Save this to self.precalc_extrapolation_nearest in test_interpolator:\n', repr(f_extrap_nearest))
    print(np.shape(x_in_full), np.shape(f_in))
    points = np.concatenate((np.reshape(x_in_full, -1)[:, np.newaxis], np.reshape(x_in_full, -1)[:, np.newaxis]), axis=1)
    cubic_2d = interp2d(x_in, y_in, f_in, kind='cubic')
    cubic_2da = RectBivariateSpline(x_in, y_in, f_in, kx=3, ky=3)
    f_cubic = cubic_2d(xsamples, ysamples)
    f_cubica = cubic_2da(xsamples, ysamples)
    print('Cubic spline at xsamples, ysamples created using. interp2d(kind=cubic)',
          'Save this to self.precalc_interpolation in test_interpolator in setup_cubic:\n', repr(f_cubic))
    from scipy.interpolate import CloughTocher2DInterpolator

    x_flat = np.reshape(x_in_full, -1)
    y_flat = np.reshape(y_in_full, -1)
    f_inflat = np.reshape(f_in, -1)
    xy_flat = np.concatenate((x_flat[:, np.newaxis], y_flat[:, np.newaxis]), axis=1)
    interp_clough_tocher = CloughTocher2DInterpolator(xy_flat, f_inflat)
    grid_z = griddata(xy_flat, f_inflat, (xsamples_in_full, ysamples_in_full), method='cubic')

    check_plot = True
    if check_plot:
        interpolator2D = Interpolator2DGrid(x_in, y_in, f_in, 'cubic', 'linear', extrapolation_range=2.0)
        import matplotlib.pyplot as plt
        from matplotlib import cm
        fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
        surf = ax[0].plot_surface(x_in_full, y_in_full, f_in, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        f_out = np.zeros((len(xsamples), len(ysamples)))
        for i in range(len(xsamples)):
            for j in range(len(ysamples)):
                f_out[i, j] = interpolator2D(xsamples[i], ysamples[j])
        f_out_extrap = np.zeros((len(xsamples_in_bounds), ))
        for i in range(len(xsamples_in_bounds)):
            f_out_extrap[i] = interpolator2D(xsamples_in_bounds[i], ysamples_in_bounds[i])
        print(np.shape(xsamples_in_full), np.shape(ysamples_in_full), np.shape(f_out))
        ax[0].scatter(xsamples_in_full, ysamples_in_full, f_out, color='r')
        print(np.shape(f_out_extrap), np.shape(xsamples_in_bounds), np.shape(ysamples_in_bounds), np.shape(f_extrap_nearest))
        ax[0].scatter(xsamples_in_bounds, ysamples_in_bounds, f_out_extrap, color='g')
        ax[0].scatter(xsamples_in_bounds, ysamples_in_bounds, f_extrap_nearest, color='m')
        print(np.shape(collapsed_xsamples_in_full), np.shape(cubic_2d(collapsed_xsamples_in_full, collapsed_ysamples_in_full)))
        ax[1].scatter(collapsed_xsamples_in_full, collapsed_ysamples_in_full, cubic_2d(xsamples, ysamples), color='m')
        f_in_2 = function_to_spline(xsamples_in_full, ysamples_in_full, factor)

        ax[1].scatter(collapsed_xsamples_in_full, collapsed_ysamples_in_full, f_in_2, color='g')
        ax[1].scatter(collapsed_xsamples_in_full, collapsed_ysamples_in_full, interp_clough_tocher(xsamples_in_full, ysamples_in_full), color='b')
        ax[1].scatter(collapsed_xsamples_in_full, collapsed_ysamples_in_full, grid_z, color='k')
        ax[1].scatter(collapsed_xsamples_in_full, collapsed_ysamples_in_full, f_cubica, color='r')
        # ax.scatter(xsamples_in_full, ysamples_in_full, f_linear, color='b')

        surf = ax[1].plot_surface(xsamples_in_full, ysamples_in_full, f_out, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        surf = ax[2].plot_surface(xsamples_in_full, ysamples_in_full, f_in_2, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        plt.show()
