
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
from raysect.core.math.function.float.function3d.interpolate.interpolator3dgrid import Interpolator3DGrid
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


def make_open_sphere(r, top_half=True):
    dr, dtheta, dphi = 0.1, np.pi / 250.0, np.pi / 250.0
    if top_half:
        [theta, phi] = np.mgrid[0.:np.pi + dtheta:dtheta, -np.pi:np.pi / 2. + dphi:dphi]
    else:
        [theta, phi] = np.mgrid[-np.pi:0. + dtheta:dtheta, -np.pi:0. / 2. + dphi:dphi]
    x, y, z = spherical_r_theta_phi_to_xyz(r, theta, phi)

    [r_range, theta_range, phi_range] = np.mgrid[0.:r:r/50., -np.pi:0. + dtheta:dtheta, np.pi / 2.:np.pi / 2. + dphi:dphi*2.]
    x_edge_low, y_edge_low, z_edge_low = spherical_r_theta_phi_to_xyz(r_range, theta_range, phi_range)
    x_edge_high, y_edge_high, z_edge_high = spherical_r_theta_phi_to_xyz(r, theta, np.pi / 2. + dphi)
    x_edge_low = np.reshape(x_edge_low, (np.shape(x_edge_low)[0], np.shape(x_edge_low)[1]))
    y_edge_low = np.reshape(y_edge_low, (np.shape(y_edge_low)[0], np.shape(y_edge_low)[1]))
    z_edge_low = np.reshape(z_edge_low, (np.shape(z_edge_low)[0], np.shape(z_edge_low)[1]))
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

        # make a surface at constant r, spherical coords
        r = 5
        X, Y, Z, X_edge_1, Y_edge_1, Z_edge_1 , X_edge_2, Y_edge_2, Z_edge_2 = make_open_sphere(r)
        F = function_to_spline(X, Y, Z, factor=factor)
        F_edge_1 = function_to_spline(X_edge_1, Y_edge_1, Z_edge_1, factor=factor)
        lower_sphere = X_UPPER
        print(np.shape(X), np.shape(Y), np.shape(Z), np.shape(X_edge_1), np.shape(Y_edge_1), np.shape(Z_edge_1))

        cmap_in = plt.get_cmap('viridis')
        fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
        min_colourmap = np.min(f_in)
        max_colourmap = np.max(f_in)
        # pcm = ax[0].pcolor(X, Y, Z, norm=[np.min(F), np.max(F)],  cmap='viridis', shading='auto')
        c_norm = SymLogNorm(vmin=min_colourmap, vmax=max_colourmap, linthresh=0.03)

        colourmap = cm.get_cmap('viridis', 512)

        face_f_edge_1_rescaled_log = c_norm(F_edge_1)
        face_f_rescaled_log = c_norm(F)
        # surf = ax[0].plot_surface(X, Y, Z, facecolors=cmap_in(face_f_rescaled_log), norm=c_norm, cmap='viridis', linewidth=0, antialiased=False, vmin=min_colourmap, vmax=max_colourmap, zorder=0)
        surf = ax[0].plot_surface(X_edge_1, Y_edge_1, Z_edge_1, facecolors=cmap_in(face_f_edge_1_rescaled_log), cmap='viridis', norm=c_norm, linewidth=0, antialiased=False, zorder=1)
        fig.colorbar(surf, ax=ax[0])
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_zlabel('z')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].set_zlabel('z')

        ax[0].set_title('Spline knots')
        ax[1].set_title('Interpolated points for testing')
        ax[2].set_title('Interpolated points for detailed view')


        plt.show()
