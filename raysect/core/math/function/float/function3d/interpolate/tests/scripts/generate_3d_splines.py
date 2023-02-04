
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

import numpy as np
from raysect.core.math.function.float.function3d.interpolate.interpolator3darray import Interpolator3DArray
from matplotlib.colors import SymLogNorm, Normalize
import scipy
import sys
from raysect.core.math.function.float.function3d.interpolate.tests.data.interpolator3d_test_data import \
    TestInterpolatorLoadBigValues, TestInterpolatorLoadNormalValues, TestInterpolatorLoadSmallValues,\
    TestInterpolatorLoadBigValuesUneven, TestInterpolatorLoadNormalValuesUneven, TestInterpolatorLoadSmallValuesUneven
from raysect.core.math.function.float.function3d.interpolate.tests.test_interpolator_3d import X_LOWER, X_UPPER,\
    NB_XSAMPLES, NB_X, X_EXTRAP_DELTA_MAX, PRECISION, Y_LOWER, Y_UPPER, NB_YSAMPLES, NB_Y, \
    Y_EXTRAP_DELTA_MAX, EXTRAPOLATION_RANGE, large_extrapolation_range, Z_LOWER, Z_UPPER, \
    NB_ZSAMPLES, NB_Z, Z_EXTRAP_DELTA_MAX, N_EXTRAPOLATION, uneven_linspace


# Force scientific format to get the right number of significant figures
np.set_printoptions(30000, linewidth=100, formatter={'float': lambda x_str: format(x_str, '.'+str(PRECISION)+'E')},
                    threshold=sys.maxsize)


# Overwrite imported values here.
VISUAL_NOT_TESTS = False
if VISUAL_NOT_TESTS:
    NB_X = 51
    NB_Y = 51
    NB_Z = 51
    NB_XSAMPLES = 101
    NB_YSAMPLES = 101
    NB_ZSAMPLES = 101

X_EXTRAP_DELTA_MIN = 0.04
Y_EXTRAP_DELTA_MIN = 0.04
Z_EXTRAP_DELTA_MIN = 0.04

BIG_VALUE_FACTOR = 20.
SMALL_VALUE_FACTOR = -20.


def docstring_test():
    """
    .. code-block:: python

        >>> from raysect.core.math.function.float.function3d.interpolate.interpolator3darray import Interpolator3DArray
        >>>
        >>> x = np.linspace(-1., 1., 20)
        >>> y = np.linspace(-1., 1., 20)
        >>> z = np.linspace(-1., 1., 20)
        >>> x_array, y_array, z_array = np.meshgrid(x, y, z, indexing='ij')
        >>> f = np.exp(-(x_array**2 + y_array**2 + z_array**2))
        >>> interpolator3D = Interpolator3DArray(x, y, z, f, 'cubic', 'nearest', 1.0, 1.0, 1.0)
        >>> # Interpolation
        >>> interpolator3D(1.0, 1.0, 0.2)
        0.1300281183136766
        >>> # Extrapolation
        >>> interpolator3D(1.0, 1.0, 1.1)
        0.0497870683678659
        >>> # Extrapolation out of bounds
        >>> interpolator3D(1.0, 1.0, 2.1)
        ValueError: The specified value (z=2.1) is outside of extrapolation range.
    """
    pass


def get_extrapolation_input_values(
        x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, x_extrap_delta_max, y_extrap_delta_max,
        z_extrap_delta_max, x_extrap_delta_min, y_extrap_delta_min, z_extrap_delta_min):
    xsamples_extrap_out_of_bounds_options = np.array(
        [x_lower - x_extrap_delta_max, (x_lower + x_upper) / 2., x_upper + x_extrap_delta_max])

    ysamples_extrap_out_of_bounds_options = np.array(
        [y_lower - y_extrap_delta_max, (y_lower + y_upper) / 2., y_upper + y_extrap_delta_max])

    zsamples_extrap_out_of_bounds_options = np.array(
        [z_lower - z_extrap_delta_max, (z_lower + z_upper) / 2., z_upper + z_extrap_delta_max])

    xsamples_extrap_in_bounds_options = np.array(
        [x_lower - x_extrap_delta_min, (x_lower + x_upper) / 2., x_upper + x_extrap_delta_min])

    ysamples_extrap_in_bounds_options = np.array(
        [y_lower - y_extrap_delta_min, (y_lower + y_upper) / 2., y_upper + y_extrap_delta_min])

    zsamples_extrap_in_bounds_options = np.array(
        [z_lower - z_extrap_delta_min, (z_lower + z_upper) / 2., z_upper + z_extrap_delta_min])

    xsamples_extrap_out_of_bounds = []
    ysamples_extrap_out_of_bounds = []
    zsamples_extrap_out_of_bounds = []
    xsamples_extrap_in_bounds = []
    ysamples_extrap_in_bounds = []
    zsamples_extrap_in_bounds = []
    edge_indicies_x = [0, len(xsamples_extrap_out_of_bounds_options) - 1]
    edge_indicies_y = [0, len(ysamples_extrap_out_of_bounds_options) - 1]
    edge_indicies_z = [0, len(zsamples_extrap_out_of_bounds_options) - 1]
    for i_x in range(len(xsamples_extrap_out_of_bounds_options)):
        for j_y in range(len(ysamples_extrap_out_of_bounds_options)):
            for k_z in range(len(zsamples_extrap_out_of_bounds_options)):
                if not (i_x not in edge_indicies_x and j_y not in edge_indicies_y and k_z not in edge_indicies_z):
                    xsamples_extrap_out_of_bounds.append(xsamples_extrap_out_of_bounds_options[i_x])
                    ysamples_extrap_out_of_bounds.append(ysamples_extrap_out_of_bounds_options[j_y])
                    zsamples_extrap_out_of_bounds.append(zsamples_extrap_out_of_bounds_options[k_z])
                    xsamples_extrap_in_bounds.append(xsamples_extrap_in_bounds_options[i_x])
                    ysamples_extrap_in_bounds.append(ysamples_extrap_in_bounds_options[j_y])
                    zsamples_extrap_in_bounds.append(zsamples_extrap_in_bounds_options[k_z])
    return \
        np.array(xsamples_extrap_out_of_bounds), np.array(ysamples_extrap_out_of_bounds), \
        np.array(zsamples_extrap_out_of_bounds), np.array(xsamples_extrap_in_bounds), \
        np.array(ysamples_extrap_in_bounds), np.array(zsamples_extrap_in_bounds)


def pcolourmesh_corners(input_array):
    return np.concatenate((input_array[:-1] - np.diff(input_array)/2.,
                           np.array([input_array[-1] - (input_array[-1] - input_array[-2]) / 2.,
                                     input_array[-1] + (input_array[-1] - input_array[-2]) / 2.])), axis=0)


def function_to_spline(x_input, y_input, z_input, factor_in):
    t = np.pi * np.sqrt((x_input ** 2 + y_input ** 2 + z_input ** 2))
    return factor_in*np.sinc(t)


if __name__ == '__main__':
    # Calculate for big values, small values, or normal values
    big_values = False
    small_values = True
    log_scale = False
    uneven_spacing = False
    use_saved_datastore_spline_knots = True
    verbose_options = [False, True, False, False]
    if VISUAL_NOT_TESTS:
        index_x_in = 40
    else:
        index_x_in = 4
    index_y_in = 0
    index_z_in = 0
    index_y_plot = 0
    index_z_plot = 0
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
        z_in = uneven_linspace(Z_LOWER, Z_UPPER, NB_Z, offset_fraction=1./3.)
    else:
        x_in = np.linspace(X_LOWER, X_UPPER, NB_X)
        y_in = np.linspace(Y_LOWER, Y_UPPER, NB_Y)
        z_in = np.linspace(Z_LOWER, Z_UPPER, NB_Z)

    x_in_full, y_in_full, z_in_full = np.meshgrid(x_in, y_in, z_in, indexing='ij')
    f_in = function_to_spline(x_in_full, y_in_full, z_in_full, factor)

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
    if verbose_options[0]:
        print('Save this to self.data in test_interpolator:\n', repr(f_in))

    xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
    ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)
    zsamples = np.linspace(Z_LOWER, Z_UPPER, NB_ZSAMPLES)

    xsamples_extrapolation, ysamples_extrapolation, zsamples_extrapolation = large_extrapolation_range(
        xsamples, ysamples, zsamples, EXTRAPOLATION_RANGE, N_EXTRAPOLATION
    )

    # # Extrapolation x and y values
    xsamples_out_of_bounds, ysamples_out_of_bounds, zsamples_out_of_bounds, xsamples_in_bounds,  ysamples_in_bounds, \
    zsamples_in_bounds = get_extrapolation_input_values(
        X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, Z_LOWER, Z_UPPER, X_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MAX,
        Z_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN, Y_EXTRAP_DELTA_MIN, Z_EXTRAP_DELTA_MIN
        )

    interpolator3D = Interpolator3DArray(x_in, y_in, z_in, f_in, 'linear', 'linear', extrapolation_range_x=2.0,
                                         extrapolation_range_y=2.0, extrapolation_range_z=2.0)
    if VISUAL_NOT_TESTS:
        n_lower_upper_interp = 51
    else:
        n_lower_upper_interp = 19
    n_lower = 50
    lower_p = 0.9
    xsamples_lower_and_upper = np.linspace(X_LOWER, X_UPPER, n_lower_upper_interp)
    ysamples_lower_and_upper = np.linspace(Y_LOWER, Y_UPPER, n_lower_upper_interp)
    zsamples_lower_and_upper = np.linspace(Z_LOWER, Z_UPPER, n_lower_upper_interp)
    xsamples_lower_and_upper = np.concatenate((np.linspace(X_LOWER - (X_UPPER - X_LOWER) * lower_p, X_LOWER, n_lower)[
                                               :-1], xsamples_lower_and_upper,
                                               np.linspace(X_UPPER, X_UPPER + (X_UPPER - X_LOWER) * lower_p, n_lower)[
                                               1:]))
    ysamples_lower_and_upper = np.concatenate((np.linspace(Y_LOWER - (Y_UPPER - Y_LOWER) * lower_p, Y_LOWER, n_lower)[
                                               :-1], ysamples_lower_and_upper,
                                               np.linspace(Y_UPPER, Y_UPPER + (Y_UPPER - Y_LOWER) * lower_p, n_lower)[
                                               1:]))
    zsamples_lower_and_upper = np.concatenate((np.linspace(Z_LOWER - (Z_UPPER - Z_LOWER) * lower_p, Z_LOWER, n_lower)[
                                               :-1], zsamples_lower_and_upper,
                                               np.linspace(Z_UPPER, Z_UPPER + (Z_UPPER - Z_LOWER) * lower_p, n_lower)[
                                               1:]))
    index_ysamples_lower_upper = np.where(x_in[index_y_in] == ysamples_lower_and_upper)[0].item()

    # extrapolation to save
    f_extrapolation_output = np.zeros((len(xsamples_extrapolation), ))
    for i in range(len(xsamples_extrapolation)):
        f_extrapolation_output[i] = interpolator3D(
            xsamples_extrapolation[i], ysamples_extrapolation[i], zsamples_extrapolation[i]
        )
    if verbose_options[1]:
        print('Output of extrapolation to be saved:\n', repr(f_extrapolation_output))

    check_plot = True
    if check_plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        # Install mayavi and pyQt5

        main_plots_on = True
        if main_plots_on:
            fig, ax = plt.subplots(1, 4)
            fig1, ax1 = plt.subplots(1, 2)
            if not (x_in[index_x_in] == xsamples).any():
                raise ValueError(
                    f'To compare a slice, NB_XSAMPLES={NB_XSAMPLES}-1, NB_YSAMPLES={NB_YSAMPLES}-1, NB_ZSAMPLES='
                    f'{NB_ZSAMPLES}-1 must be divisible by NB_X={NB_X}-1, NB_Y={NB_Y}-1, NB_Z={NB_Z}-1'
                )
            if not (y_in[index_y_in] == ysamples_lower_and_upper).any():
                raise ValueError(
                    f'To compare a slice, NB_XSAMPLES={NB_XSAMPLES}-1, NB_YSAMPLES={NB_YSAMPLES}-1, NB_ZSAMPLES='
                    f'{NB_ZSAMPLES}-1 must be divisible by NB_X={NB_X}-1, NB_Y={NB_Y}-1, NB_Z={NB_Z}-1'
                )
            index_xsamples = np.where(x_in[index_x_in] == xsamples)[0].item()
            index_ysamples_lower_upper = np.where(y_in[index_y_in] == ysamples_lower_and_upper)[0].item()
            # index_ysamples_lower_upper = 0
            # index_zsamples_lower_upper = 0
            index_zsamples_lower_upper = np.where(z_in[index_z_in] == zsamples_lower_and_upper)[0].item()
            f_plot_x = f_in[index_x_in, :, :]

            y_corners_x = pcolourmesh_corners(y_in)
            z_corners_x = pcolourmesh_corners(z_in)

            min_colourmap = np.min(f_in)
            max_colourmap = np.max(f_in)
            if log_scale:
                c_norm = SymLogNorm(vmin=min_colourmap, vmax=max_colourmap, linthresh=0.03)
            else:
                c_norm = Normalize(vmin=min_colourmap, vmax=max_colourmap)
            colourmap = cm.get_cmap('viridis', 512)

            ax[0].pcolormesh(y_corners_x, z_corners_x, f_plot_x, norm=c_norm, cmap='viridis')
            # ax[0].pcolormesh(y_in, z_in, f_plot_x)
            ax[0].set_aspect('equal')

            f_out = np.zeros((len(xsamples), len(ysamples), len(zsamples)))
            for i in range(len(xsamples)):
                for j in range(len(ysamples)):
                    for k in range(len(zsamples)):
                        f_out[i, j, k] = interpolator3D(xsamples[i], ysamples[j], zsamples[k])
            if verbose_options[2]:
                print('Test interpolation:\n', repr(f_out))

            f_out_lower_and_upper = np.zeros((len(xsamples_lower_and_upper), len(ysamples_lower_and_upper),
                                              len(zsamples_lower_and_upper)))
            for i in range(len(xsamples_lower_and_upper)):
                for j in range(len(ysamples_lower_and_upper)):
                    for k in range(len(zsamples_lower_and_upper)):
                        f_out_lower_and_upper[i, j, k] = interpolator3D(
                            xsamples_lower_and_upper[i], ysamples_lower_and_upper[j], zsamples_lower_and_upper[k]
                        )

            f_out_extrapolation = np.zeros((len(xsamples_extrapolation), ))
            for i in range(len(xsamples_extrapolation)):
                f_out_extrapolation[i] = interpolator3D(
                    xsamples_extrapolation[i], ysamples_extrapolation[i], zsamples_extrapolation[i]
                )
            if verbose_options[3]:
                print('New output of extrapolation to be saved:\n', repr(f_out_extrapolation))

            index_xsamples_extrap = np.where(x_in[index_x_in] == xsamples_extrapolation)
            f_out_x_extrapolation = f_out_extrapolation[index_xsamples_extrap]

            im = ax[3].scatter(
                ysamples_extrapolation[index_xsamples_extrap], zsamples_extrapolation[index_xsamples_extrap],
                c=f_out_x_extrapolation, norm=c_norm, cmap='viridis', s=10
            )
            ax[3].set_aspect('equal')

            f_out_x = f_out[index_xsamples, :, :]

            ysamples_mesh, zsamples_mesh = np.meshgrid(ysamples, zsamples)
            ax[0].scatter(
                ysamples_mesh.ravel(), zsamples_mesh.ravel(), c=f_out_x.ravel(), norm=c_norm, cmap='viridis', s=10
            )
            index_y_print = -1
            index_z_print = 0

            index_ysamples_print = np.where(y_in[index_y_print] == ysamples)[0].item()
            index_zsamples_print = np.where(z_in[index_z_print] == zsamples)[0].item()
            ax[0].set_title('Slice of x', size=20)
            ax[1].set_title(f'Interpolated points \nin slice of x={x_in[index_x_in]}', size=20)

            y_corners_xsamples = pcolourmesh_corners(ysamples)
            z_corners_xsamples = pcolourmesh_corners(zsamples)

            im2 = ax[1].pcolormesh(y_corners_xsamples, z_corners_xsamples, f_out_x, norm=c_norm, cmap='viridis')

            ax[1].set_aspect('equal')
            if not (x_in[index_x_in] == xsamples_lower_and_upper).any():
                raise ValueError(
                    f'To compare a slice, n_lower_upper={n_lower}-1, must be divisible by NB_X={NB_X}-1, NB_Y={NB_Y}-1,'
                    f' NB_Z={NB_Z}-1'
                )
            index_xsamples_lower_and_upper = np.where(x_in[index_x_in] == xsamples_lower_and_upper)[0].item()

            y_corners_xsamples_lower_and_upper = pcolourmesh_corners(ysamples_lower_and_upper)
            z_corners_xsamples_lower_and_upper = pcolourmesh_corners(zsamples_lower_and_upper)
            f_out_lower_and_upper_x = f_out_lower_and_upper[index_xsamples_lower_and_upper, :, :]
            im3 = ax[2].pcolormesh(
                y_corners_xsamples_lower_and_upper, z_corners_xsamples_lower_and_upper, f_out_lower_and_upper_x,
                norm=c_norm, cmap='viridis'
            )

            check_array_z = np.zeros(len(zsamples_lower_and_upper))
            check_array_y = np.zeros(len(ysamples_lower_and_upper))
            for i in range(len(zsamples_lower_and_upper)):
                check_array_z[i] = interpolator3D(
                    x_in[index_x_in], ysamples_lower_and_upper[index_ysamples_lower_upper], zsamples_lower_and_upper[i]
                )
                check_array_y[i] = interpolator3D(
                    x_in[index_x_in], ysamples_lower_and_upper[i], zsamples_lower_and_upper[index_zsamples_lower_upper]
                )

            ax1[0].plot(zsamples_lower_and_upper, f_out_lower_and_upper_x[index_ysamples_lower_upper, :])
            ax1[0].plot(z_in, f_in[index_x_in, index_y_in, :], 'bo')
            ax1[0].plot(zsamples_lower_and_upper, check_array_z, 'gx')
            ax1[1].plot(ysamples_lower_and_upper, check_array_y)
            # ax1[1].plot(ysamples_lower_and_upper, f_out_lower_and_upper_x[:, index_z_plot])
            ax1[0].axvline(z_in[0], color='r', linestyle='--')
            ax1[0].axvline(z_in[-1], color='r', linestyle='--')
            ax1[1].axvline(y_in[0], color='r', linestyle='--')
            ax1[1].axvline(y_in[-1], color='r', linestyle='--')
            fig.colorbar(im, ax=ax[0])
            fig.colorbar(im2, ax=ax[1])
            fig.colorbar(im3, ax=ax[2])
            ax[2].set_aspect('equal')

            plt.show()
