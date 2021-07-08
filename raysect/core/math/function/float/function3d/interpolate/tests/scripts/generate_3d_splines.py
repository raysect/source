
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
from raysect.core.math.function.float.function3d.interpolate.interpolator3darray import Interpolator3DArray
from matplotlib.colors import ListedColormap, LogNorm, SymLogNorm
import scipy


X_LOWER = -1.0
X_UPPER = 1.0
Y_LOWER = -1.0
Y_UPPER = 1.0
Z_LOWER = -1.0
Z_UPPER = 1.0
X_EXTRAP_DELTA_MAX = 0.08
X_EXTRAP_DELTA_MIN = 0.04
Y_EXTRAP_DELTA_MAX = 0.08
Y_EXTRAP_DELTA_MIN = 0.04
Z_EXTRAP_DELTA_MAX = 0.08
Z_EXTRAP_DELTA_MIN = 0.04

NB_X = 10
NB_Y = 10
NB_Z = 10
NB_XSAMPLES = 19
NB_YSAMPLES = 19
NB_ZSAMPLES = 19

EXTRAPOLATION_RANGE = 0.06

PRECISION = 12

BIG_VALUE_FACTOR = 20.
SMALL_VALUE_FACTOR = -20.


# Force scientific format to get the right number of significant figures
np.set_printoptions(30000, linewidth=100, formatter={'float': lambda x_str: format(x_str, '.'+str(PRECISION)+'E')})


# def get_extrapolation_input_values(
#         x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, x_extrap_delta_max, y_extrap_delta_max, z_extrap_delta_max, x_extrap_delta_min, y_extrap_delta_min, z_extrap_delta_min):
#     xsamples_extrap_out_of_bounds_options = np.array(
#         [x_lower - x_extrap_delta_max, (x_lower + x_upper) / 2., x_upper + x_extrap_delta_max])
#
#     ysamples_extrap_out_of_bounds_options = np.array(
#         [y_lower - y_extrap_delta_max, (y_lower + y_upper) / 2., y_upper + y_extrap_delta_max])
#
#     zsamples_extrap_out_of_bounds_options = np.array(
#         [z_lower - z_extrap_delta_max, (z_lower + z_upper) / 2., z_upper + z_extrap_delta_max])
#
#     xsamples_extrap_in_bounds_options = np.array(
#         [x_lower - x_extrap_delta_min, (x_lower + x_upper) / 2., x_upper + x_extrap_delta_min])
#
#     ysamples_extrap_in_bounds_options = np.array(
#         [y_lower - y_extrap_delta_min, (y_lower + y_upper) / 2., y_upper + y_extrap_delta_min])
#
#     zsamples_extrap_in_bounds_options = np.array(
#         [z_lower - z_extrap_delta_min, (z_lower + z_upper) / 2., z_upper + z_extrap_delta_min])
#
#     xsamples_extrap_out_of_bounds = []
#     ysamples_extrap_out_of_bounds = []
#     xsamples_extrap_in_bounds = []
#     ysamples_extrap_in_bounds = []
#     edge_indicies = [0, len(xsamples_extrap_out_of_bounds_options) - 1]
#     for i_x in range(len(xsamples_extrap_out_of_bounds_options)):
#         for j_y in range(len(xsamples_extrap_out_of_bounds_options)):
#             if not (i_x not in edge_indicies and j_y not in edge_indicies):
#                 xsamples_extrap_out_of_bounds.append(xsamples_extrap_out_of_bounds_options[i_x])
#                 ysamples_extrap_out_of_bounds.append(ysamples_extrap_out_of_bounds_options[j_y])
#                 xsamples_extrap_in_bounds.append(xsamples_extrap_in_bounds_options[i_x])
#                 ysamples_extrap_in_bounds.append(ysamples_extrap_in_bounds_options[j_y])
#     return \
#         np.array(xsamples_extrap_out_of_bounds), np.array(ysamples_extrap_out_of_bounds), \
#         np.array(xsamples_extrap_in_bounds), np.array(ysamples_extrap_in_bounds)


def pcolourmesh_corners(input_array):
    return np.linspace(input_array[0] - (input_array[1] - input_array[0]) / 2., input_array[-1] + (input_array[-1] - input_array[-2]) / 2., len(input_array) + 1)


def function_to_spline(x_input, y_input, z_input, factor):
    t = np.pi * np.sqrt((x_input ** 2 + y_input ** 2 + z_input ** 2))
    return factor*np.sinc(t)


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
    z_in = np.linspace(Z_LOWER, Z_UPPER, NB_Z)
    x_in_full, y_in_full = np.meshgrid(x_in, y_in)
    x_in_full, y_in_full, z_in_full = np.meshgrid(x_in, y_in, z_in, indexing='ij')
    f_in = function_to_spline(x_in_full, y_in_full, z_in_full, factor)

    print('Save this to self.data in test_interpolator:\n', repr(f_in))

    xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
    ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)
    zsamples = np.linspace(Z_LOWER, Z_UPPER, NB_ZSAMPLES)

    # # Extrapolation x and y values
    # xsamples_out_of_bounds, ysamples_out_of_bounds, zsamples_out_of_bounds, xsamples_in_bounds,  ysamples_in_bounds,  zsamples_in_bounds = \
    #     get_extrapolation_input_values(
    #         X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, Z_LOWER, Z_UPPER, X_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MAX, Z_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN,
    #         Y_EXTRAP_DELTA_MIN, Z_EXTRAP_DELTA_MIN
    #     )


    check_plot = True
    if check_plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        # Install mayavi and pyQt5
        n_lower_upper_interp = 19
        n_lower = 5
        lower_p = 0.9
        xsamples_lower_and_upper = np.linspace(X_LOWER, X_UPPER, n_lower_upper_interp)
        ysamples_lower_and_upper = np.linspace(Y_LOWER, Y_UPPER, n_lower_upper_interp)
        zsamples_lower_and_upper = np.linspace(Z_LOWER, Z_UPPER, n_lower_upper_interp)
        xsamples_lower_and_upper = np.concatenate((np.linspace(X_LOWER - (X_UPPER - X_LOWER)*lower_p, X_LOWER, n_lower)[:-1], xsamples_lower_and_upper, np.linspace(X_UPPER, X_UPPER + (X_UPPER - X_LOWER)*lower_p, n_lower)[1:]))
        ysamples_lower_and_upper = np.concatenate((np.linspace(Y_LOWER - (Y_UPPER - Y_LOWER)*lower_p, Y_LOWER, n_lower)[:-1], ysamples_lower_and_upper, np.linspace(Y_UPPER, Y_UPPER + (Y_UPPER - Y_LOWER)*lower_p, n_lower)[1:]))
        zsamples_lower_and_upper = np.concatenate((np.linspace(Z_LOWER - (Z_UPPER - Z_LOWER)*lower_p, Z_LOWER, n_lower)[:-1], zsamples_lower_and_upper, np.linspace(Z_UPPER, Z_UPPER + (Z_UPPER - Z_LOWER)*lower_p, n_lower)[1:]))

        interpolator3D = Interpolator3DArray(x_in, y_in, z_in, f_in, 'cubic', 'linear', extrapolation_range_x=2.0, extrapolation_range_y=2.0, extrapolation_range_z=2.0)
        main_plots_on = True
        mayavi_plots_on = False
        if main_plots_on:
            fig, ax = plt.subplots(1, 3)
            index_x_in = 5
            if not (x_in[index_x_in] == xsamples).any():
                raise ValueError(
                    f'To compare a slice, NB_XSAMPLES={NB_XSAMPLES}-1, NB_YSAMPLES={NB_YSAMPLES}-1, NB_ZSAMPLES='
                    f'{NB_ZSAMPLES}-1 must be divisible by NB_X={NB_X}-1, NB_Y={NB_Y}-1, NB_Z={NB_Z}-1'
                )
            index_xsamples = np.where(x_in[index_x_in] == xsamples)[0].item()
            f_plot_x = f_in[index_x_in, :, :]

            y_corners_x = pcolourmesh_corners(y_in)
            z_corners_x = pcolourmesh_corners(z_in)

            min_colourmap = np.min(f_in)
            max_colourmap = np.max(f_in)
            c_norm = SymLogNorm(vmin=min_colourmap, vmax=max_colourmap, linthresh=0.03)
            colourmap = cm.get_cmap('viridis', 512)

            ax[0].pcolormesh(y_corners_x, z_corners_x, f_plot_x, norm=c_norm, cmap='viridis')
            # ax[0].pcolormesh(y_in, z_in, f_plot_x)
            ax[0].set_aspect('equal')

            f_out = np.zeros((len(xsamples), len(ysamples), len(zsamples)))
            for i in range(len(xsamples)):
                for j in range(len(ysamples)):
                    for k in range(len(zsamples)):
                        f_out[i, j, k] = interpolator3D(xsamples[i], ysamples[j], zsamples[k])
            f_out_lower_and_upper = np.zeros((len(xsamples_lower_and_upper), len(ysamples_lower_and_upper), len(zsamples_lower_and_upper)))
            for i in range(len(xsamples_lower_and_upper)):
                for j in range(len(ysamples_lower_and_upper)):
                    for k in range(len(zsamples_lower_and_upper)):
                        f_out_lower_and_upper[i, j, k] = interpolator3D(xsamples_lower_and_upper[i], ysamples_lower_and_upper[j], zsamples_lower_and_upper[k])
            f_out_x = f_out[index_xsamples, :, :]
            ysamples_mesh, zsamples_mesh = np.meshgrid(ysamples, zsamples)
            im = ax[0].scatter(ysamples_mesh.ravel(), zsamples_mesh.ravel(), c=f_out_x.ravel(), norm=c_norm, cmap='viridis', s=10)
            index_y_print = -1
            index_z_print = 0

            index_ysamples_print = np.where(y_in[index_y_print] == ysamples)[0].item()
            index_zsamples_print = np.where(z_in[index_z_print] == zsamples)[0].item()
            ax[0].set_title('Slice of x', size=20)
            ax[1].set_title('Interpolated points in slice of x', size=20)

            y_corners_xsamples = pcolourmesh_corners(ysamples)
            z_corners_xsamples = pcolourmesh_corners(zsamples)

            im2 = ax[1].pcolormesh(y_corners_xsamples, z_corners_xsamples, f_out_x, norm=c_norm, cmap='viridis')

            ax[1].set_aspect('equal')
            print(x_in[index_x_in] == xsamples_lower_and_upper)
            if not (x_in[index_x_in] == xsamples_lower_and_upper).any():
                raise ValueError(
                    f'To compare a slice, n_lower_upper={n_lower}-1, must be divisible by NB_X={NB_X}-1, NB_Y={NB_Y}-1, NB_Z={NB_Z}-1'
                )
            index_xsamples_lower_and_upper = np.where(x_in[index_x_in] == xsamples_lower_and_upper)[0].item()

            y_corners_xsamples_lower_and_upper = pcolourmesh_corners(ysamples_lower_and_upper)
            z_corners_xsamples_lower_and_upper = pcolourmesh_corners(zsamples_lower_and_upper)
            f_out_lower_and_upper_x = f_out_lower_and_upper[index_xsamples_lower_and_upper, :, :]
            im3 = ax[2].pcolormesh(y_corners_xsamples_lower_and_upper, z_corners_xsamples_lower_and_upper, f_out_lower_and_upper_x, norm=c_norm, cmap='viridis')


            fig.colorbar(im, ax=ax[0])
            fig.colorbar(im2, ax=ax[1])
            fig.colorbar(im3, ax=ax[2])
            ax[2].set_aspect('equal')

            plt.show()
        if mayavi_plots_on:
            from mayavi import mlab
            # https://docs.enthought.com/mayavi/mayavi/mlab_case_studies.html
            x, y, z = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
            n = 20

            s = function_to_spline(x, y, z, 1.)
            # mlab.pipeline.volume(mlab.pipeline.scalar_field(s))        # mlab.volume_slice(s, plane_orientation='x_axes', slice_index=10)
            src = mlab.pipeline.scalar_field(x, y, z, s)
            mlab.pipeline.iso_surface(src, contours=[s.min() + 0.1 * s.ptp(), ], opacity=0.1, colormap='viridis')
            mlab.pipeline.iso_surface(src, contours=[s.max() - 0.1 * s.ptp(), ], colormap='viridis')
            mlab.pipeline.image_plane_widget(src,
                                             plane_orientation='z_axes',
                                             slice_index=10, colormap='viridis'
                                             )
            mlab.axes()
            mlab.orientation_axes()
            dxyz = 0.1
            x_point, y_point, z_point = np.mgrid[-0.5:-0.5+dxyz:4j, -0.5:-0.5+dxyz:4j, -0.5:-0.5+dxyz:4j]

            x_point = np.array([-0.5, 0.0])
            y_point = np.array([-0.5, 0.0])
            z_point = np.array([0.0, 0.0])
            s_point = np.array([2000., 0.001])
            mlab.points3d(x_point, y_point, z_point, s_point, colormap="viridis", scale_mode='none')
            x_point = 0.5
            y_point = 0.5
            z_point = 0.0
            s_point = 1
            x_point = np.array([-0.5])
            y_point = np.array([-0.5])
            z_point = np.array([0.0])
            s_point = np.array([10])
            ones = np.ones(1)
            scalars = np.arange(1)  # Key point: set an integer for each point
            # mlab.pipeline.scalar_scatter(x_point, y_point, z_point, scalar=function_to_spline(x_point, y_point, z_point, factor=1.), colormap='viridis')
            # pts = mlab.quiver3d(x_point, y_point, z_point, ones, ones, ones, scalars=s_point, mode='sphere', scale_factor=1)  # Create points
            # pts.glyph.color_mode = 'color_by_scalar'  # Color by scalar
            # mlab.points3d(x_point, y_point, z_point, s_point, colormap="viridis")

            mlab.show()

