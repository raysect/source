
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
NB_XSAMPLES = 13
NB_YSAMPLES = 13
NB_ZSAMPLES = 13

EXTRAPOLATION_RANGE = 0.06

PRECISION = 12

BIG_VALUE_FACTOR = 20.
SMALL_VALUE_FACTOR = -20.


# Force scientific format to get the right number of significant figures
np.set_printoptions(30000, linewidth=100, formatter={'float': lambda x_str: format(x_str, '.'+str(PRECISION)+'E')})


def function_to_spline(x_input, y_input, z_input, factor):
    t = np.pi * np.sqrt((x_input ** 2 + y_input ** 2 + z_input ** 2))
    return factor*np.sinc(t)


def make_open_sphere(r):
    dr, dtheta, dphi = 0.1, np.pi / 250.0, np.pi / 250.0
    [theta, phi] = np.mgrid[0.:np.pi + dtheta:dtheta, -np.pi:np.pi / 2. + dphi:dphi]
    x, y, z = spherical_r_theta_phi_to_xyz(r, theta, phi)

    [r_range_low, theta_range_low, phi_range_low] = np.mgrid[0.:r:r/50., -np.pi:0. + dtheta:dtheta, -np.pi / 2.:-np.pi / 2. + dphi:dphi*2.]
    [r_range_high, theta_range_high, phi_range_high] = np.mgrid[0.:r:r/50., -np.pi:0. + dtheta:dtheta, np.pi:np.pi + dphi:dphi*2.]
    x_edge_low, y_edge_low, z_edge_low = spherical_r_theta_phi_to_xyz(r_range_low, theta_range_low, phi_range_low)
    x_edge_high, y_edge_high, z_edge_high = spherical_r_theta_phi_to_xyz(r_range_high, theta_range_high, phi_range_high)
    x_edge_low = np.reshape(x_edge_low, (np.shape(x_edge_low)[0], np.shape(x_edge_low)[1]))
    y_edge_low = np.reshape(y_edge_low, (np.shape(y_edge_low)[0], np.shape(y_edge_low)[1]))
    z_edge_low = np.reshape(z_edge_low, (np.shape(z_edge_low)[0], np.shape(z_edge_low)[1]))
    x_edge_high = np.reshape(x_edge_high, (np.shape(x_edge_high)[0], np.shape(x_edge_high)[1]))
    y_edge_high = np.reshape(y_edge_high, (np.shape(y_edge_high)[0], np.shape(y_edge_high)[1]))
    z_edge_high = np.reshape(z_edge_high, (np.shape(z_edge_high)[0], np.shape(z_edge_high)[1]))
    return x, y, z, x_edge_low, y_edge_low, z_edge_low, x_edge_high, y_edge_high, z_edge_high


def spherical_r_theta_phi_to_xyz(r, theta, phi):
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return x, y, z


def spherical_xyz_to_r_theta_phi(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z/r)
    return r, theta, phi


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





    check_plot = True
    if check_plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        # Install mayavi and pyQt5
        from mayavi import mlab
        interpolator3D = Interpolator3DArray(x_in, y_in, z_in, f_in, 'linear', 'none', extrapolation_range_x=2.0, extrapolation_range_y=2.0, extrapolation_range_z=2.0)

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
        print(x)
        mlab.show()
        plt.show()
