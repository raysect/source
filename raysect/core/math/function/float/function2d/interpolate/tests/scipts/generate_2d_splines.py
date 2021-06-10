
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


X_LOWER = -1.0
X_UPPER = 1.0
Y_LOWER = -1.0
Y_UPPER = 1.0
Y_EXTRAP_DELTA_MAX = 0.08
Y_EXTRAP_DELTA_MIN = 0.04

NB_X = 10
NB_Y = 10
NB_XSAMPLES = 30
NB_YSAMPLES = 30

EXTRAPOLATION_RANGE = 0.06

PRECISION = 12

BIG_VALUE_FACTOR = 20.
SMALL_VALUE_FACTOR = -20.


def function_to_spline(x_input, y_input, factor):
    t = np.pi * np.sqrt((x_input ** 2 + y_input ** 2))
    return factor*np.sinc(t)


x_in = np.linspace(X_LOWER, X_UPPER, NB_X)
y_in = np.linspace(Y_LOWER, Y_UPPER, NB_Y)
x_in_full, y_in_full = np.meshgrid(x_in, y_in)
xy_in = np.concatenate((x_in_full[:, :, np.newaxis], y_in_full[:, :, np.newaxis]), axis=2)
f_in = function_to_spline(xy_in[:, :, 0], xy_in[:, :, 1], 1.)

# Make the sampled points between spline knots and find the precalc_interpolation used in test_interpolator.setup_cubic
xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)


# Tempory measure for not extrapolating
xsamples = xsamples[:-1]
ysamples = ysamples[:-1]

# Make grid
xsamples_in_full, ysamples_in_full = np.meshgrid(xsamples, ysamples)

check_plot = True
if check_plot:
    interpolator2D = Interpolator2DGrid(x_in, y_in, f_in, 'linear', 'none', extrapolation_range=2.0)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x_in_full, y_in_full, f_in, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    f_out = np.zeros((len(xsamples), len(ysamples)))
    for i in range(len(xsamples)):
        for j in range(len(ysamples)):
            f_out[i, j] = interpolator2D(xsamples[i], ysamples[j])
    print(np.shape(xsamples_in_full), np.shape(ysamples_in_full), np.shape(f_out))
    ax.scatter(xsamples_in_full, ysamples_in_full, f_out, color='r')
    plt.show()
